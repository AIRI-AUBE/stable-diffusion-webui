from __future__ import annotations

import os
import sys
import time
import importlib
import signal
import re
import warnings
import json
from threading import Thread
from typing import Iterable

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from packaging import version

import logging

logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

from modules import paths, timer, import_hook, errors  # noqa: F401

startup_timer = timer.Timer()

import torch
import pytorch_lightning   # noqa: F401 # pytorch_lightning should be imported after torch, but it re-enables warnings on import so import once to disable them
warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="pytorch_lightning")
warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")


startup_timer.record("import torch")

import gradio
startup_timer.record("import gradio")

import ldm.modules.encoders.modules  # noqa: F401
startup_timer.record("import ldm")

from modules import extra_networks
from modules.call_queue import wrap_gradio_gpu_call, wrap_queued_call, queue_lock  # noqa: F401

# Truncate version number of nightly/local build of PyTorch to not cause exceptions with CodeFormer or Safetensors
if ".dev" in torch.__version__ or "+git" in torch.__version__:
    torch.__long_version__ = torch.__version__
    torch.__version__ = re.search(r'[\d.]+[\d]', torch.__version__).group(0)

from modules import shared, sd_samplers, upscaler, extensions, localization, ui_tempdir, ui_extra_networks, config_states
import modules.codeformer_model as codeformer
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.scripts
import modules.sd_hijack
import modules.sd_hijack_optimizations
import modules.sd_models
import modules.sd_vae
import modules.txt2img
import modules.script_callbacks
import modules.textual_inversion.textual_inversion
import modules.progress

import modules.ui
from modules import modelloader
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork


from huggingface_hub import hf_hub_download
import boto3
import json
import shutil
import traceback
from modules.sync_models import initial_s3_download,sync_s3_folder


if cmd_opts.train:
    from botocore.exceptions import ClientError
    from extensions.sd_dreambooth_extension.dreambooth.db_config import DreamboothConfig
    from extensions.sd_dreambooth_extension.scripts.dreambooth import start_training_from_config, create_model
    from extensions.sd_dreambooth_extension.scripts.dreambooth import performance_wizard, training_wizard
    from extensions.sd_dreambooth_extension.dreambooth.db_concept import Concept
    from modules import paths

startup_timer.record("other imports")


if cmd_opts.server_name:
    server_name = cmd_opts.server_name
else:
    server_name = "0.0.0.0" if cmd_opts.listen else None


def fix_asyncio_event_loop_policy():
    """
        The default `asyncio` event loop policy only automatically creates
        event loops in the main threads. Other threads must create event
        loops explicitly or `asyncio.get_event_loop` (and therefore
        `.IOLoop.current`) will fail. Installing this policy allows event
        loops to be created automatically on any thread, matching the
        behavior of Tornado versions prior to 5.0 (or 5.0 on Python 2).
    """

    import asyncio

    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        # "Any thread" and "selector" should be orthogonal, but there's not a clean
        # interface for composing policies so pick the right base.
        _BasePolicy = asyncio.WindowsSelectorEventLoopPolicy  # type: ignore
    else:
        _BasePolicy = asyncio.DefaultEventLoopPolicy

    class AnyThreadEventLoopPolicy(_BasePolicy):  # type: ignore
        """Event loop policy that allows loop creation on any thread.
        Usage::

            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
        """

        def get_event_loop(self) -> asyncio.AbstractEventLoop:
            try:
                return super().get_event_loop()
            except (RuntimeError, AssertionError):
                # This was an AssertionError in python 3.4.2 (which ships with debian jessie)
                # and changed to a RuntimeError in 3.4.3.
                # "There is no current event loop in thread %r"
                loop = self.new_event_loop()
                self.set_event_loop(loop)
                return loop

    asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())


def check_versions():
    if shared.cmd_opts.skip_version_check:
        return

    expected_torch_version = "2.0.0"

    if version.parse(torch.__version__) < version.parse(expected_torch_version):
        errors.print_error_explanation(f"""
You are running torch {torch.__version__}.
The program is tested to work with torch {expected_torch_version}.
To reinstall the desired version, run with commandline flag --reinstall-torch.
Beware that this will cause a lot of large files to be downloaded, as well as
there are reports of issues with training tab on the latest version.

Use --skip-version-check commandline argument to disable this check.
        """.strip())

    expected_xformers_version = "0.0.17"
    if shared.xformers_available:
        import xformers

        if version.parse(xformers.__version__) < version.parse(expected_xformers_version):
            errors.print_error_explanation(f"""
You are running xformers {xformers.__version__}.
The program is tested to work with xformers {expected_xformers_version}.
To reinstall the desired version, run with commandline flag --reinstall-xformers.

Use --skip-version-check commandline argument to disable this check.
            """.strip())


def restore_config_state_file():
    config_state_file = shared.opts.restore_config_state_file
    if config_state_file == "":
        return

    shared.opts.restore_config_state_file = ""
    shared.opts.save(shared.config_filename)

    if os.path.isfile(config_state_file):
        print(f"*** About to restore extension state from file: {config_state_file}")
        with open(config_state_file, "r", encoding="utf-8") as f:
            config_state = json.load(f)
            config_states.restore_extension_config(config_state)
        startup_timer.record("restore extension config")
    elif config_state_file:
        print(f"!!! Config state backup not found: {config_state_file}")


def validate_tls_options():
    if not (cmd_opts.tls_keyfile and cmd_opts.tls_certfile):
        return

    try:
        if not os.path.exists(cmd_opts.tls_keyfile):
            print("Invalid path to TLS keyfile given")
        if not os.path.exists(cmd_opts.tls_certfile):
            print(f"Invalid path to TLS certfile: '{cmd_opts.tls_certfile}'")
    except TypeError:
        cmd_opts.tls_keyfile = cmd_opts.tls_certfile = None
        print("TLS setup invalid, running webui without TLS")
    else:
        print("Running with TLS")
    startup_timer.record("TLS")


def get_gradio_auth_creds() -> Iterable[tuple[str, ...]]:
    """
    Convert the gradio_auth and gradio_auth_path commandline arguments into
    an iterable of (username, password) tuples.
    """
    def process_credential_line(s) -> tuple[str, ...] | None:
        s = s.strip()
        if not s:
            return None
        return tuple(s.split(':', 1))

    if cmd_opts.gradio_auth:
        for cred in cmd_opts.gradio_auth.split(','):
            cred = process_credential_line(cred)
            if cred:
                yield cred

    if cmd_opts.gradio_auth_path:
        with open(cmd_opts.gradio_auth_path, 'r', encoding="utf8") as file:
            for line in file.readlines():
                for cred in line.strip().split(','):
                    cred = process_credential_line(cred)
                    if cred:
                        yield cred


def configure_sigint_handler():
    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    if not os.environ.get("COVERAGE_RUN"):
        # Don't install the immediate-quit handler when running under coverage,
        # as then the coverage report won't be generated.
        signal.signal(signal.SIGINT, sigint_handler)


def configure_opts_onchange():
    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights()), call=False)
    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("sd_vae_as_default", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("temp_dir", ui_tempdir.on_tmpdir_changed)
    shared.opts.onchange("gradio_theme", shared.reload_gradio_theme)
    shared.opts.onchange("cross_attention_optimization", wrap_queued_call(lambda: modules.sd_hijack.model_hijack.redo_hijack(shared.sd_model)), call=False)
    startup_timer.record("opts onchange")


def initialize():
    fix_asyncio_event_loop_policy()
    validate_tls_options()
    configure_sigint_handler()
    check_versions()
    modelloader.cleanup_models()
    configure_opts_onchange()

    modules.sd_models.setup_model()
    startup_timer.record("setup SD model")

    codeformer.setup_model(cmd_opts.codeformer_models_path)
    startup_timer.record("setup codeformer")

    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    startup_timer.record("setup gfpgan")

    initialize_rest(reload_script_modules=False)


def initialize_rest(*, reload_script_modules=False):
    """
    Called both from initialize() and when reloading the webui.
    """
    sd_samplers.set_samplers()
    extensions.list_extensions()
    startup_timer.record("list extensions")

    restore_config_state_file()

    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        modules.scripts.load_scripts()
        return

    modules.sd_models.list_models()
    startup_timer.record("list SD models")

    localization.list_localizations(cmd_opts.localizations_dir)

    modules.scripts.load_scripts()
    startup_timer.record("load scripts")

    if reload_script_modules:
        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)
        startup_timer.record("reload script modules")

    modelloader.load_upscalers()
    startup_timer.record("load upscalers")

    modules.sd_vae.refresh_vae_list()
    startup_timer.record("refresh VAE")
    modules.textual_inversion.textual_inversion.list_textual_inversion_templates()
    startup_timer.record("refresh textual inversion templates")

    modules.script_callbacks.on_list_optimizers(modules.sd_hijack_optimizations.list_optimizers)
    modules.sd_hijack.list_optimizers()
    startup_timer.record("scripts list_optimizers")

    def load_model():
        """
        Accesses shared.sd_model property to load model.
        After it's available, if it has been loaded before this access by some extension,
        its optimization may be None because the list of optimizaers has neet been filled
        by that time, so we apply optimization again.
        """

        shared.sd_model  # noqa: B018

        if modules.sd_hijack.current_optimizer is None:
            modules.sd_hijack.apply_optimizations()

    Thread(target=load_model).start()

    shared.reload_hypernetworks()
    startup_timer.record("reload hypernetworks")

    ui_extra_networks.initialize()
    ui_extra_networks.register_default_pages()

    extra_networks.initialize()
    extra_networks.register_default_extra_networks()
    startup_timer.record("initialize extra networks")


def setup_middleware(app):
    app.middleware_stack = None  # reset current middleware to allow modifying user provided list
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    configure_cors_middleware(app)
    app.build_middleware_stack()  # rebuild middleware stack on-the-fly


def configure_cors_middleware(app):
    cors_options = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "allow_credentials": True,
    }
    if cmd_opts.cors_allow_origins:
        cors_options["allow_origins"] = cmd_opts.cors_allow_origins.split(',')
    if cmd_opts.cors_allow_origins_regex:
        cors_options["allow_origin_regex"] = cmd_opts.cors_allow_origins_regex
    app.add_middleware(CORSMiddleware, **cors_options)


def create_api(app):
    from modules.api.api import Api
    api = Api(app, queue_lock)
    return api


def api_only():
    initialize()

    app = FastAPI()
    setup_middleware(app)
    api = create_api(app)

    modules.script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1", port=cmd_opts.port if cmd_opts.port else 7861)


def stop_route(request):
    shared.state.server_command = "stop"
    return Response("Stopping.")


def webui():
    global cache

    launch_api = cmd_opts.api

    if launch_api:
        models_config_s3uri = os.environ.get('models_config_s3uri', None)
        if models_config_s3uri:
            bucket, key = shared.get_bucket_and_key(models_config_s3uri)
            s3_object = shared.s3_client.get_object(Bucket=bucket, Key=key)
            bytes = s3_object["Body"].read()
            payload = bytes.decode('utf8')
            huggingface_models = json.loads(payload).get('huggingface_models', None)
            s3_models = json.loads(payload).get('s3_models', None)
            http_models = json.loads(payload).get('http_models', None)
        else:
            huggingface_models = os.environ.get('huggingface_models', None)
            huggingface_models = json.loads(huggingface_models) if huggingface_models else None
            s3_models = os.environ.get('s3_models', None)
            s3_models = json.loads(s3_models) if s3_models else None
            http_models = os.environ.get('http_models', None)
            http_models = json.loads(http_models) if http_models else None

        if huggingface_models:
            for huggingface_model in huggingface_models:
                repo_id = huggingface_model['repo_id']
                filename = huggingface_model['filename']
                name = huggingface_model['name']

                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=f'/tmp/models/{name}',
                    cache_dir='/tmp/cache/huggingface'
                )

        if s3_models:
            for s3_model in s3_models:
                uri = s3_model['uri']
                name = s3_model['name']
                shared.s3_download(uri, f'/tmp/models/{name}')

        if http_models:
            for http_model in http_models:
                uri = http_model['uri']
                filename = http_model['filename']
                name = http_model['name']
                shared.http_download(uri, f'/tmp/models/{name}/{filename}')

        print(os.system('df -h'))
        sd_models_tmp_dir = f"{shared.tmp_models_dir}/Stable-diffusion/"
        cn_models_tmp_dir = f"{shared.tmp_models_dir}/ControlNet/"
        lora_models_tmp_dir = f"{shared.tmp_models_dir}/Lora/"
        cache_dir = f"{shared.tmp_cache_dir}/"
        session = boto3.Session()
        region_name = session.region_name
        sts_client = session.client('sts')
        account_id = sts_client.get_caller_identity()['Account']
        sg_s3_bucket = f"sagemaker-{region_name}-{account_id}"
        if not shared.models_s3_bucket:
            shared.models_s3_bucket = os.environ['sg_default_bucket'] if os.environ.get('sg_default_bucket') else sg_s3_bucket
            shared.s3_folder_sd = "stable-diffusion-webui/models/Stable-diffusion"
            shared.s3_folder_cn = "stable-diffusion-webui/models/ControlNet"
            shared.s3_folder_lora = "stable-diffusion-webui/models/Lora"
        #only download the cn models and the first sd model from default bucket, to accerlate the startup time
        initial_s3_download(shared.s3_client, shared.s3_folder_sd, sd_models_tmp_dir,cache_dir,'sd')
        sync_s3_folder(sd_models_tmp_dir, cache_dir, 'sd')
        sync_s3_folder(cn_models_tmp_dir, cache_dir, 'cn')
        sync_s3_folder(lora_models_tmp_dir, cache_dir, 'lora')
    initialize()

    while 1:
        if shared.opts.clean_temp_dir_at_start:
            ui_tempdir.cleanup_tmpdr()
            startup_timer.record("cleanup temp dir")

        modules.script_callbacks.before_ui_callback()
        startup_timer.record("scripts before_ui_callback")

        shared.demo = modules.ui.create_ui()
        startup_timer.record("create ui")

        if not cmd_opts.no_gradio_queue:
            shared.demo.queue(64)

        gradio_auth_creds = list(get_gradio_auth_creds()) or None

        # this restores the missing /docs endpoint
        if launch_api and not hasattr(FastAPI, 'original_setup'):
            # TODO: replace this with `launch(app_kwargs=...)` if https://github.com/gradio-app/gradio/pull/4282 gets merged
            def fastapi_setup(self):
                self.docs_url = "/docs"
                self.redoc_url = "/redoc"
                self.original_setup()

            FastAPI.original_setup = FastAPI.setup
            FastAPI.setup = fastapi_setup

        app, local_url, share_url = shared.demo.launch(
            share=cmd_opts.share,
            server_name=server_name,
            server_port=cmd_opts.port,
            ssl_keyfile=cmd_opts.tls_keyfile,
            ssl_certfile=cmd_opts.tls_certfile,
            ssl_verify=cmd_opts.disable_tls_verify,
            debug=cmd_opts.gradio_debug,
            auth=gradio_auth_creds,
            inbrowser=cmd_opts.autolaunch,
            prevent_thread_lock=True,
            allowed_paths=cmd_opts.gradio_allowed_path,
        )
        if cmd_opts.add_stop_route:
            app.add_route("/_stop", stop_route, methods=["POST"])

        # after initial launch, disable --autolaunch for subsequent restarts
        cmd_opts.autolaunch = False

        startup_timer.record("gradio launch")

        # gradio uses a very open CORS policy via app.user_middleware, which makes it possible for
        # an attacker to trick the user into opening a malicious HTML page, which makes a request to the
        # running web ui and do whatever the attacker wants, including installing an extension and
        # running its code. We disable this here. Suggested by RyotaK.
        app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']

        setup_middleware(app)

        modules.progress.setup_progress_api(app)
        modules.ui.setup_ui_api(app)

        if launch_api:
            create_api(app)

        ui_extra_networks.add_pages_to_demo(app)

        modules.script_callbacks.app_started_callback(shared.demo, app)
        startup_timer.record("scripts app_started_callback")

        print(f"Startup time: {startup_timer.summary()}.")

        if cmd_opts.subpath:
            redirector = FastAPI()
            redirector.get("/")
            gradio.mount_gradio_app(redirector, shared.demo, path=f"/{cmd_opts.subpath}")

        try:
            while True:
                server_command = shared.state.wait_for_server_command(timeout=5)
                if server_command:
                    if server_command in ("stop", "restart"):
                        break
                    else:
                        print(f"Unknown server command: {server_command}")
        except KeyboardInterrupt:
            print('Caught KeyboardInterrupt, stopping...')
            server_command = "stop"

        if server_command == "stop":
            print("Stopping server...")
            # If we catch a keyboard interrupt, we want to stop the server and exit.
            shared.demo.close()
            break
        print('Restarting UI...')
        shared.demo.close()
        time.sleep(0.5)
        startup_timer.reset()
        modules.script_callbacks.app_reload_callback()
        startup_timer.record("app reload callback")
        modules.script_callbacks.script_unloaded_callback()
        startup_timer.record("scripts unloaded callback")
        initialize_rest(reload_script_modules=True)

        modules.script_callbacks.on_list_optimizers(modules.sd_hijack_optimizations.list_optimizers)
        modules.sd_hijack.list_optimizers()
        startup_timer.record("scripts list_optimizers")


        modules.script_callbacks.model_loaded_callback(shared.sd_model)
        startup_timer.record("model loaded callback")

        modelloader.load_upscalers()
        startup_timer.record("load upscalers")

        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)
        startup_timer.record("reload script modules")

        modules.sd_models.list_models()
        startup_timer.record("list SD models")

        shared.reload_hypernetworks()
        startup_timer.record("reload hypernetworks")

        ui_extra_networks.intialize()
        ui_extra_networks.register_page(ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion())
        ui_extra_networks.register_page(ui_extra_networks_hypernets.ExtraNetworksPageHypernetworks())
        ui_extra_networks.register_page(ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints())

        extra_networks.initialize()
        extra_networks.register_extra_network(extra_networks_hypernet.ExtraNetworkHypernet())
        startup_timer.record("initialize extra networks")

if cmd_opts.train:
    def train():
        initialize()

        train_args = json.loads(cmd_opts.train_args)

        sd_models_s3uri = cmd_opts.sd_models_s3uri
        db_models_s3uri = cmd_opts.db_models_s3uri
        lora_models_s3uri = cmd_opts.lora_models_s3uri

        db_create_new_db_model = train_args['train_dreambooth_settings']['db_create_new_db_model']
        db_use_txt2img = train_args['train_dreambooth_settings']['db_use_txt2img']
        db_train_wizard_person = train_args['train_dreambooth_settings']['db_train_wizard_person']
        db_train_wizard_object = train_args['train_dreambooth_settings']['db_train_wizard_object']
        db_performance_wizard = train_args['train_dreambooth_settings']['db_performance_wizard']

        if db_create_new_db_model:
            db_new_model_name = train_args['train_dreambooth_settings']['db_new_model_name']
            db_new_model_src = train_args['train_dreambooth_settings']['db_new_model_src']
            db_new_model_scheduler = train_args['train_dreambooth_settings']['db_new_model_scheduler']
            db_create_from_hub = train_args['train_dreambooth_settings']['db_create_from_hub']
            db_new_model_url = train_args['train_dreambooth_settings']['db_new_model_url']
            db_new_model_token = train_args['train_dreambooth_settings']['db_new_model_token']
            db_new_model_extract_ema = train_args['train_dreambooth_settings']['db_new_model_extract_ema']
            db_train_unfrozen = train_args['train_dreambooth_settings']['db_train_unfrozen']
            db_512_model = train_args['train_dreambooth_settings']['db_512_model']
            db_save_safetensors = train_args['train_dreambooth_settings']['db_save_safetensors']

            db_model_name, db_model_path, db_revision, db_epochs, db_scheduler, db_src, db_has_ema, db_v2, db_resolution = create_model(
                db_new_model_name,
                db_new_model_src,
                db_new_model_scheduler,
                db_create_from_hub,
                db_new_model_url,
                db_new_model_token,
                db_new_model_extract_ema,
                db_train_unfrozen,
                db_512_model
            )
            dreambooth_config_id = cmd_opts.dreambooth_config_id
            try:
                with open(f'/opt/ml/input/data/config/{dreambooth_config_id}.json', 'r') as f:
                    content = f.read()
            except Exception:
                content = None

            if content:
                params_dict = json.loads(content)

                params_dict['db_model_name'] = db_model_name
                params_dict['db_model_path'] = db_model_path
                params_dict['db_revision'] = db_revision
                params_dict['db_epochs'] = db_epochs
                params_dict['db_scheduler'] = db_scheduler
                params_dict['db_src'] = db_src
                params_dict['db_has_ema'] = db_has_ema
                params_dict['db_v2'] = db_v2
                params_dict['db_resolution'] = db_resolution

                if db_train_wizard_person or db_train_wizard_object:
                    db_num_train_epochs, \
                    c1_num_class_images_per, \
                    c2_num_class_images_per, \
                    c3_num_class_images_per, \
                    c4_num_class_images_per = training_wizard(db_train_wizard_person if db_train_wizard_person else db_train_wizard_object)

                    params_dict['db_num_train_epochs'] = db_num_train_epochs
                    params_dict['c1_num_class_images_per'] = c1_num_class_images_per
                    params_dict['c1_num_class_images_per'] = c2_num_class_images_per
                    params_dict['c1_num_class_images_per'] = c3_num_class_images_per
                    params_dict['c1_num_class_images_per'] = c4_num_class_images_per
                if db_performance_wizard:
                    attention, \
                    gradient_checkpointing, \
                    gradient_accumulation_steps, \
                    mixed_precision, \
                    cache_latents, \
                    sample_batch_size, \
                    train_batch_size, \
                    stop_text_encoder, \
                    use_8bit_adam, \
                    use_lora, \
                    use_ema, \
                    save_samples_every, \
                    save_weights_every = performance_wizard()

                    params_dict['attention'] = attention
                    params_dict['gradient_checkpointing'] = gradient_checkpointing
                    params_dict['gradient_accumulation_steps'] = gradient_accumulation_steps
                    params_dict['mixed_precision'] = mixed_precision
                    params_dict['cache_latents'] = cache_latents
                    params_dict['sample_batch_size'] = sample_batch_size
                    params_dict['train_batch_size'] = train_batch_size
                    params_dict['stop_text_encoder'] = stop_text_encoder
                    params_dict['use_8bit_adam'] = use_8bit_adam
                    params_dict['use_lora'] = use_lora
                    params_dict['use_ema'] = use_ema
                    params_dict['save_samples_every'] = save_samples_every
                    params_dict['params_dict'] = save_weights_every

                db_config = DreamboothConfig(db_model_name)
                concept_keys = ["c1_", "c2_", "c3_", "c4_"]
                concepts_list = []
                # If using a concepts file/string, keep concepts_list empty.
                if params_dict["db_use_concepts"] and params_dict["db_concepts_path"]:
                    concepts_list = []
                    params_dict["concepts_list"] = concepts_list
                else:
                    for concept_key in concept_keys:
                        concept_dict = {}
                        for key, param in params_dict.items():
                            if concept_key in key and param is not None:
                                concept_dict[key.replace(concept_key, "")] = param
                        concept_test = Concept(concept_dict)
                        if concept_test.is_valid:
                            concepts_list.append(concept_test.__dict__)
                    existing_concepts = params_dict["concepts_list"] if "concepts_list" in params_dict else []
                    if len(concepts_list) and not len(existing_concepts):
                        params_dict["concepts_list"] = concepts_list

                db_config.load_params(params_dict)
        else:
            db_model_name = train_args['train_dreambooth_settings']['db_model_name']
            db_config = DreamboothConfig(db_model_name)

        print(vars(db_config))
        start_training_from_config(
            db_config,
            db_use_txt2img,
        )

        cmd_sd_models_path = cmd_opts.ckpt_dir
        sd_models_dir = os.path.join(shared.models_path, "Stable-diffusion")
        if cmd_sd_models_path is not None:
            sd_models_dir = cmd_sd_models_path

        try:
            cmd_dreambooth_models_path = cmd_opts.dreambooth_models_path
        except:
            cmd_dreambooth_models_path = None

        try:
            cmd_lora_models_path = shared.cmd_opts.lora_models_path
        except:
            cmd_lora_models_path = None

        db_model_dir = os.path.dirname(cmd_dreambooth_models_path) if cmd_dreambooth_models_path else paths.models_path
        db_model_dir = os.path.join(db_model_dir, "dreambooth")

        lora_model_dir = os.path.dirname(cmd_lora_models_path) if cmd_lora_models_path else paths.models_path
        lora_model_dir = os.path.join(lora_model_dir, "lora")

        try:
            print('Uploading SD Models...')
            shared.upload_s3files(
                sd_models_s3uri,
                os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.yaml')
            )
            if db_save_safetensors:
                shared.upload_s3files(
                    sd_models_s3uri,
                    os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.safetensors')
                )
            else:
                shared.upload_s3files(
                    sd_models_s3uri,
                    os.path.join(sd_models_dir, db_model_name, f'{db_model_name}_*.ckpt')
                )
            print('Uploading DB Models...')
            shared.upload_s3folder(
                f'{db_models_s3uri}{db_model_name}',
                os.path.join(db_model_dir, db_model_name)
            )
            if db_config.use_lora:
                print('Uploading Lora Models...')
                shared.upload_s3files(
                    lora_models_s3uri,
                    os.path.join(lora_model_dir, f'{db_model_name}_*.pt')
                )
            os.makedirs(os.path.dirname("/opt/ml/model/"), exist_ok=True)
            os.makedirs(os.path.dirname("/opt/ml/model/Stable-diffusion/"), exist_ok=True)
            train_steps=int(db_config.revision)
            model_file_basename = f'{db_model_name}_{train_steps}_lora' if db_config.use_lora else f'{db_model_name}_{train_steps}'
            f1=os.path.join(sd_models_dir, db_model_name, f'{model_file_basename}.yaml')
            if os.path.exists(f1):
                shutil.copy(f1,"/opt/ml/model/Stable-diffusion/")
            if db_save_safetensors:
                f2=os.path.join(sd_models_dir, db_model_name, f'{model_file_basename}.safetensors')
                if os.path.exists(f2):
                    shutil.copy(f2,"/opt/ml/model/Stable-diffusion/")
            else:
                f2=os.path.join(sd_models_dir, db_model_name, f'{model_file_basename}.ckpt')
                if os.path.exists(f2):
                    shutil.copy(f2,"/opt/ml/model/Stable-diffusion/")
        except Exception as e:
            traceback.print_exc()
            print(e)

if __name__ == "__main__":
    if cmd_opts.train:
        train()
    elif cmd_opts.nowebui:
        api_only()
    else:
        webui()
