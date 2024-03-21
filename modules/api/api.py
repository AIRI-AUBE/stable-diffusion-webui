import base64
import io
import os
import time
import copy
import datetime
import uvicorn
import ipaddress
import requests
import gradio as gr
from threading import Lock
from io import BytesIO
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from secrets import compare_digest

import modules.shared as shared
from modules import sd_samplers, deepbooru, sd_hijack, images, scripts, ui, postprocessing, errors, restart, shared_items
from modules.api import models
from modules.shared import opts
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images
from modules.textual_inversion.textual_inversion import create_embedding, train_embedding
from modules.textual_inversion.preprocess import preprocess
from modules.hypernetworks.hypernetwork import create_hypernetwork, train_hypernetwork
from PIL import PngImagePlugin,Image
from modules.sd_models import unload_model_weights, reload_model_weights, checkpoint_aliases
from modules.sd_models_config import find_checkpoint_config_near_filename
from modules.realesrgan_model import get_realesrgan_models
from modules import devices
from typing import Dict, List, Any
import piexif
import piexif.helper
from contextlib import closing

from typing import Union
import traceback
from modules.sd_vae import reload_vae_weights, refresh_vae_list
import uuid
import json
import requests

def script_name_to_index(name, scripts):
    try:
        return [script.title().lower() for script in scripts].index(name.lower())
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Script '{name}' not found") from e


def validate_sampler_name(name):
    config = sd_samplers.all_samplers_map.get(name, None)
    if config is None:
        raise HTTPException(status_code=404, detail="Sampler not found")

    return name


def setUpscalers(req: dict):
    reqDict = vars(req)
    reqDict['extras_upscaler_1'] = reqDict.pop('upscaler_1', None)
    reqDict['extras_upscaler_2'] = reqDict.pop('upscaler_2', None)
    return reqDict


def verify_url(url):
    """Returns True if the url refers to a global resource."""

    import socket
    from urllib.parse import urlparse
    try:
        parsed_url = urlparse(url)
        domain_name = parsed_url.netloc
        host = socket.gethostbyname_ex(domain_name)
        for ip in host[2]:
            ip_addr = ipaddress.ip_address(ip)
            if not ip_addr.is_global:
                return False
    except Exception:
        return False

    return True

def decode_to_image(encoding):
    image = None
    try:
        if encoding.startswith("http://") or encoding.startswith("https://"):
            response = requests.get(encoding)
            if response.status_code == 200:
                encoding = response.text
                image = Image.open(BytesIO(response.content))
        elif encoding.startswith("s3://"):
            bucket, key = shared.get_bucket_and_key(encoding)
            response = shared.s3_client.get_object(
                Bucket=bucket,
                Key=key
            )
            image = Image.open(response['Body'])
        else:
            if encoding.startswith("data:image/"):
                encoding = encoding.split(";")[1].split(",")[1]
            image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as err:
        raise HTTPException(status_code=500, detail="Invalid encoded image")

def mask_decode_to_image(encoding):
    image = decode_to_image(encoding)
    print(f"mask_decode_to_image image before {type(image)}")
    try:
        # response = requests.get('http://0.0.0.0:8080/sam/sam-model')
        # print(f'\nsam/sam-model: {response.text}\n')
        response = requests.get('http://0.0.0.0:8080/sam/heartbeat')
        print(f'\nsam/heartbeat: {response.text}\n')
        if "Success" in response.text:
            # print(f'\nsam/heartbeat: {response.text}\n')
            image = encode_pil_to_base64(image)
            print(f"mask_decode_to_image image mid 1 {type(image)}")
            image = image.decode('utf-8')  # Decode bytes to string
            print(f"mask_decode_to_image image mid 2 {type(image)}")
            dilate_value = 16
            start_time = time.time()
            response_raw = requests.post('http://0.0.0.0:8080/sam/dilate-mask', json={
                "input_image": image,
                "mask": image,
                "dilate_amount": dilate_value
            }, timeout=60)
            response = response_raw.json()
            print(f"mask_decode_to_image sam dilate-mask response 1 took {time.time() - start_time}s.")
            if "mask" in response:
                print(f"mask_decode_to_image response 2 mask image length {len(response['mask'])}")
                print(f"mask_decode_to_image image mid 3 {type(response['mask'])}")
                image = decode_base64_to_image(response['mask'])
                print(f"mask_decode_to_image image after {type(image)}")
                print(f'SAM successfully dilated mask by {dilate_value}.')

            else:
                print(f'!!!! Error: SAM did not return a masked_image! response is {response_raw}')
                raise HTTPException(status_code=500, detail="Error: sam did not return a masked_image!")
        else:
            print(f'!!!! Error: SAM heartbeat lost!')
    except Exception as err:
        print(f'!!!! Error: SAM exception {err}!!!!')
        pass

    return image

def decode_base64_to_image(encoding):
    if encoding.startswith("http://") or encoding.startswith("https://"):
        if not opts.api_enable_requests:
            raise HTTPException(status_code=500, detail="Requests not allowed")

        if opts.api_forbid_local_requests and not verify_url(encoding):
            raise HTTPException(status_code=500, detail="Request to local resource not allowed")

        headers = {'user-agent': opts.api_useragent} if opts.api_useragent else {}
        response = requests.get(encoding, timeout=30, headers=headers)
        try:
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            raise HTTPException(status_code=500, detail="Invalid image url") from e

    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    try:
        image = Image.open(BytesIO(base64.b64decode(encoding)))
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail="Invalid encoded image") from e


user_input_data = {}


def set_img_exif_dict(image_id="img_id_1"):
    # print(f"log@{datetime.datetime.now().strftime(f'%Y%m%d%H%M%S')} ***&&& set_img_exif_dict Writing exif")
    title = "AIRI"
    date_taken = "2001:01:01 01:01:01"
    global user_input_data
    if "date_taken" in user_input_data:
        date_taken = user_input_data['date_taken']
    copyright = "© AIRI Lab. All Rights Reserved."
    camera_maker = "AIRI Lab"
    camera_model = "AIRI Model 1.0"
    user_id = "AIRI tester"
    if "user_id" in user_input_data:
        user_id = user_input_data['user_id']
    generation_id = "gen_id_1"
    if "generation_id" in user_input_data:
        generation_id = user_input_data['generation_id']
    keywords = f"Generated in AIRI platform. airilab.com. Generation ID: {generation_id}, Image ID: {image_id}"
    title = f"{user_id}_airilab.com_{image_id}" #####
    description = f"An image processed by the AIRI platform. airilab.com. Generation ID: {generation_id}, Image ID: {image_id}"
    software = "AIRI Platform v1.0"
    # imagenum = "imagenum?"
    # seed = "seed?"
    exif_dict = {
        "0th": {
            piexif.ImageIFD.ImageDescription: title.encode('utf-8'),
            piexif.ImageIFD.Make: camera_maker.encode('utf-8'),
            piexif.ImageIFD.Model: camera_model.encode('utf-8'),
            # piexif.ImageIFD.Copyright: copyright.encode('utf-8'), #decision to remove
            piexif.ImageIFD.Artist: user_id.encode('utf-8'),
            piexif.ImageIFD.ProcessingSoftware: software.encode('utf-8'),
            piexif.ImageIFD.Software: software.encode('utf-8'),
            piexif.ImageIFD.DateTime: date_taken.encode('utf-8'),
            piexif.ImageIFD.HostComputer: software.encode('utf-8'),
            # piexif.ImageIFD.ImageID: imageid.encode('utf-8'), #bad
            # piexif.ImageIFD.ImageNumber: imagenum.encode('utf-8'), #bad
            piexif.ImageIFD.ImageHistory: keywords.encode('utf-8'),
            # piexif.ImageIFD.ImageResources: description.encode('utf-8'),#bad
            # piexif.ImageIFD.Noise: seed.encode('utf-8'),#bad
            piexif.ImageIFD.Predictor: camera_model.encode('utf-8'),
            piexif.ImageIFD.OriginalRawFileData: keywords.encode('utf-8'),
            # piexif.ImageIFD.OriginalRawFileName: imageid.encode('utf-8'),#bad
            piexif.ImageIFD.ProfileCopyright: copyright.encode('utf-8'),
            piexif.ImageIFD.ProfileEmbedPolicy: software.encode('utf-8'),
            piexif.ImageIFD.Rating: "5".encode('utf-8'),
            piexif.ImageIFD.ProfileName: user_id.encode('utf-8'),
            # piexif.ImageIFD.XPAuthor: user_id.encode('utf-8'),#bad
            # piexif.ImageIFD.XPTitle: title.encode('utf-8'),#bad
            # piexif.ImageIFD.XPKeywords: keywords.encode('utf-8'),#bad
            # piexif.ImageIFD.XPComment: description.encode('utf-8'),#bad
            # piexif.ImageIFD.XPSubject: copyright.encode('utf-8'),#bad
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: date_taken.encode('utf-8'),
            piexif.ExifIFD.CameraOwnerName: user_id.encode('utf-8'),
            piexif.ExifIFD.DateTimeDigitized: date_taken.encode('utf-8'),
            piexif.ExifIFD.DeviceSettingDescription: camera_model.encode('utf-8'),
            piexif.ExifIFD.FileSource: keywords.encode('utf-8'),
            # piexif.ExifIFD.ImageUniqueID: imageid.encode('utf-8'),#bad
            piexif.ExifIFD.LensMake: camera_maker.encode('utf-8'),
            piexif.ExifIFD.LensModel: camera_model.encode('utf-8'),
            piexif.ExifIFD.MakerNote: description.encode('utf-8'),
            piexif.ExifIFD.UserComment: description.encode('utf-8'),
        }
    }

    # def print_nested_dict(nested_dict, indent=0):
    #     for key, value in nested_dict.items():
    #         print('\t' * indent + str(key))
    #         if isinstance(value, dict):
    #             print_nested_dict(value, indent + 1)
    #         else:
    #             print('\t' * (indent + 1) + str(value))

    # print_nested_dict(exif_dict)
    return exif_dict


def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:

        if opts.samples_format.lower() == 'png':
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=opts.jpeg_quality)

        elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
            if image.mode == "RGBA":
                image = image.convert("RGB")
            # parameters = image.info.get('parameters', None)
            # exif_bytes = piexif.dump({
            #     "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") }
            # })

            # Convert dict to bytes
            exif_bytes = piexif.dump(set_img_exif_dict())

            if opts.samples_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG", exif=exif_bytes, quality=opts.jpeg_quality)
                # print(f"log@{datetime.datetime.now().strftime(f'%Y%m%d%H%M%S')} ***&&& encode_pil_to_base64")
                # print(Image.open(output_bytes).getexif())  # Print the EXIF data
            else:
                image.save(output_bytes, format="WEBP", exif=exif_bytes, quality=opts.jpeg_quality)

        else:
            raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)

def export_pil_to_bytes(image, quality, image_id=""):
    with io.BytesIO() as output_bytes:

        if opts.samples_format.lower() == 'png':
            use_metadata = False
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
                    use_metadata = True
            image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=quality if quality else opts.jpeg_quality)

        elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
            # parameters = image.info.get('parameters', None)
            # exif_bytes = piexif.dump({
            #     "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") }
            # })
            # Convert dict to bytes
            exif_bytes = piexif.dump(set_img_exif_dict(image_id=image_id))

            if opts.samples_format.lower() in ("jpg", "jpeg"):
                image.save(output_bytes, format="JPEG", exif = exif_bytes, quality=quality if quality else opts.jpeg_quality)
            else:
                image.save(output_bytes, format="WEBP", exif = exif_bytes, quality=quality if quality else opts.jpeg_quality)

        else:
            raise HTTPException(status_code=500, detail="Invalid image format")

        bytes_data = output_bytes.getvalue()

    return bytes_data

def api_middleware(app: FastAPI):
    rich_available = False
    try:
        if os.environ.get('WEBUI_RICH_EXCEPTIONS', None) is not None:
            import anyio  # importing just so it can be placed on silent list
            import starlette  # importing just so it can be placed on silent list
            from rich.console import Console
            console = Console()
            rich_available = True
    except Exception:
        pass

    @app.middleware("http")
    async def log_and_time(req: Request, call_next):
        ts = time.time()
        res: Response = await call_next(req)
        duration = str(round(time.time() - ts, 4))
        res.headers["X-Process-Time"] = duration
        endpoint = req.scope.get('path', 'err')
        if shared.cmd_opts.api_log and endpoint.startswith('/sdapi'):
            print('API {t} {code} {prot}/{ver} {method} {endpoint} {cli} {duration}'.format(
                t=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                code=res.status_code,
                ver=req.scope.get('http_version', '0.0'),
                cli=req.scope.get('client', ('0:0.0.0', 0))[0],
                prot=req.scope.get('scheme', 'err'),
                method=req.scope.get('method', 'err'),
                endpoint=endpoint,
                duration=duration,
            ))
        return res

    def handle_exception(request: Request, e: Exception):
        err = {
            "error": type(e).__name__,
            "detail": vars(e).get('detail', ''),
            "body": vars(e).get('body', ''),
            "errors": str(e),
        }
        if not isinstance(e, HTTPException):  # do not print backtrace on known httpexceptions
            message = f"API error: {request.method}: {request.url} {err}"
            if rich_available:
                print(message)
                console.print_exception(show_locals=True, max_frames=2, extra_lines=1, suppress=[anyio, starlette], word_wrap=False, width=min([console.width, 200]))
            else:
                errors.report(message, exc_info=True)
        return JSONResponse(status_code=vars(e).get('status_code', 500), content=jsonable_encoder(err))

    @app.middleware("http")
    async def exception_handling(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            return handle_exception(request, e)

    @app.exception_handler(Exception)
    async def fastapi_exception_handler(request: Request, e: Exception):
        return handle_exception(request, e)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, e: HTTPException):
        return handle_exception(request, e)


class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock):
        if shared.cmd_opts.api_auth:
            self.credentials = {}
            for auth in shared.cmd_opts.api_auth.split(","):
                user, password = auth.split(":")
                self.credentials[user] = password

        self.router = APIRouter()
        self.app = app
        self.queue_lock = queue_lock
        self.invocations_lock = Lock()
        api_middleware(self.app)
        self.add_api_route("/sdapi/v1/txt2img", self.text2imgapi, methods=["POST"], response_model=models.TextToImageResponse)
        self.add_api_route("/sdapi/v1/img2img", self.img2imgapi, methods=["POST"], response_model=models.ImageToImageResponse)
        self.add_api_route("/sdapi/v1/extra-single-image", self.extras_single_image_api, methods=["POST"], response_model=models.ExtrasSingleImageResponse)
        self.add_api_route("/sdapi/v1/extra-batch-images", self.extras_batch_images_api, methods=["POST"], response_model=models.ExtrasBatchImagesResponse)
        self.add_api_route("/sdapi/v1/png-info", self.pnginfoapi, methods=["POST"], response_model=models.PNGInfoResponse)
        self.add_api_route("/sdapi/v1/progress", self.progressapi, methods=["GET"], response_model=models.ProgressResponse)
        self.add_api_route("/sdapi/v1/interrogate", self.interrogateapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/interrupt", self.interruptapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/skip", self.skip, methods=["POST"])
        self.add_api_route("/sdapi/v1/options", self.get_config, methods=["GET"], response_model=models.OptionsModel)
        self.add_api_route("/sdapi/v1/options", self.set_config, methods=["POST"])
        self.add_api_route("/sdapi/v1/cmd-flags", self.get_cmd_flags, methods=["GET"], response_model=models.FlagsModel)
        self.add_api_route("/sdapi/v1/samplers", self.get_samplers, methods=["GET"], response_model=List[models.SamplerItem])
        self.add_api_route("/sdapi/v1/upscalers", self.get_upscalers, methods=["GET"], response_model=List[models.UpscalerItem])
        self.add_api_route("/sdapi/v1/latent-upscale-modes", self.get_latent_upscale_modes, methods=["GET"], response_model=List[models.LatentUpscalerModeItem])
        self.add_api_route("/sdapi/v1/sd-models", self.get_sd_models, methods=["GET"], response_model=List[models.SDModelItem])
        self.add_api_route("/sdapi/v1/sd-vae", self.get_sd_vaes, methods=["GET"], response_model=List[models.SDVaeItem])
        self.add_api_route("/sdapi/v1/hypernetworks", self.get_hypernetworks, methods=["GET"], response_model=List[models.HypernetworkItem])
        self.add_api_route("/sdapi/v1/face-restorers", self.get_face_restorers, methods=["GET"], response_model=List[models.FaceRestorerItem])
        self.add_api_route("/sdapi/v1/realesrgan-models", self.get_realesrgan_models, methods=["GET"], response_model=List[models.RealesrganItem])
        self.add_api_route("/sdapi/v1/prompt-styles", self.get_prompt_styles, methods=["GET"], response_model=List[models.PromptStyleItem])
        self.add_api_route("/sdapi/v1/embeddings", self.get_embeddings, methods=["GET"], response_model=models.EmbeddingsResponse)
        self.add_api_route("/sdapi/v1/refresh-checkpoints", self.refresh_checkpoints, methods=["POST"])
        self.add_api_route("/sdapi/v1/refresh-vae", self.refresh_vae, methods=["POST"])
        self.add_api_route("/sdapi/v1/create/embedding", self.create_embedding, methods=["POST"], response_model=models.CreateResponse)
        self.add_api_route("/sdapi/v1/create/hypernetwork", self.create_hypernetwork, methods=["POST"], response_model=models.CreateResponse)
        self.add_api_route("/sdapi/v1/preprocess", self.preprocess, methods=["POST"], response_model=models.PreprocessResponse)
        self.add_api_route("/sdapi/v1/train/embedding", self.train_embedding, methods=["POST"], response_model=models.TrainResponse)
        self.add_api_route("/sdapi/v1/train/hypernetwork", self.train_hypernetwork, methods=["POST"], response_model=models.TrainResponse)
        self.add_api_route("/sdapi/v1/memory", self.get_memory, methods=["GET"], response_model=models.MemoryResponse)
        self.add_api_route("/sdapi/v1/unload-checkpoint", self.unloadapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/reload-checkpoint", self.reloadapi, methods=["POST"])
        self.add_api_route("/sdapi/v1/scripts", self.get_scripts_list, methods=["GET"], response_model=models.ScriptsList)
        self.add_api_route("/sdapi/v1/script-info", self.get_script_info, methods=["GET"], response_model=List[models.ScriptInfo])
        self.add_api_route("/invocations", self.invocations, methods=["POST"], response_model=Any)
        self.add_api_route("/ping", self.ping, methods=["GET"], response_model=models.PingResponse)

        if shared.cmd_opts.api_server_stop:
            self.add_api_route("/sdapi/v1/server-kill", self.kill_webui, methods=["POST"])
            self.add_api_route("/sdapi/v1/server-restart", self.restart_webui, methods=["POST"])
            self.add_api_route("/sdapi/v1/server-stop", self.stop_webui, methods=["POST"])

        self.default_script_arg_txt2img = []
        self.default_script_arg_img2img = []

    def add_api_route(self, path: str, endpoint, **kwargs):
        if shared.cmd_opts.api_auth:
            return self.app.add_api_route(path, endpoint, dependencies=[Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    def auth(self, credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
        if credentials.username in self.credentials:
            if compare_digest(credentials.password, self.credentials[credentials.username]):
                return True

        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Basic"})

    def get_selectable_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None

        script_idx = script_name_to_index(script_name, script_runner.selectable_scripts)
        script = script_runner.selectable_scripts[script_idx]
        return script, script_idx

    def get_scripts_list(self):
        t2ilist = [script.name for script in scripts.scripts_txt2img.scripts if script.name is not None]
        i2ilist = [script.name for script in scripts.scripts_img2img.scripts if script.name is not None]

        return models.ScriptsList(txt2img=t2ilist, img2img=i2ilist)

    def get_script_info(self):
        res = []

        for script_list in [scripts.scripts_txt2img.scripts, scripts.scripts_img2img.scripts]:
            res += [script.api_info for script in script_list if script.api_info is not None]

        return res

    def get_script(self, script_name, script_runner):
        if script_name is None or script_name == "":
            return None, None

        script_idx = script_name_to_index(script_name, script_runner.scripts)
        return script_runner.scripts[script_idx]

    def init_default_script_args(self, script_runner):
        #find max idx from the scripts in runner and generate a none array to init script_args
        last_arg_index = 1
        for script in script_runner.scripts:
            if last_arg_index < script.args_to:
                last_arg_index = script.args_to
        # None everywhere except position 0 to initialize script args
        script_args = [None]*last_arg_index
        script_args[0] = 0

        # get default values
        with gr.Blocks(): # will throw errors calling ui function without this
            for script in script_runner.scripts:
                if script.ui(script.is_img2img):
                    ui_default_values = []
                    for elem in script.ui(script.is_img2img):
                        ui_default_values.append(elem.value)
                    script_args[script.args_from:script.args_to] = ui_default_values
        return script_args

    def init_script_args(self, request, default_script_args, selectable_scripts, selectable_idx, script_runner):
        script_args = default_script_args.copy()
        # position 0 in script_arg is the idx+1 of the selectable script that is going to be run when using scripts.scripts_*2img.run()
        if selectable_scripts:
            script_args[selectable_scripts.args_from:selectable_scripts.args_to] = request.script_args
            script_args[0] = selectable_idx + 1

        # Now check for always on scripts
        if request.alwayson_scripts:
            global user_input_data
            user_input_data = {}
            if "user_input" in request.alwayson_scripts:
                user_input_data = request.alwayson_scripts["user_input"]
                request.alwayson_scripts.pop("user_input")

            for alwayson_script_name in request.alwayson_scripts.keys():
                alwayson_script = self.get_script(alwayson_script_name, script_runner)
                if alwayson_script is None:
                    raise HTTPException(status_code=422, detail=f"always on script {alwayson_script_name} not found")
                # Selectable script in always on script param check
                if alwayson_script.alwayson is False:
                    raise HTTPException(status_code=422, detail="Cannot have a selectable script in the always on scripts params")
                # always on script with no arg should always run so you don't really need to add them to the requests
                if "args" in request.alwayson_scripts[alwayson_script_name]:
                    # min between arg length in scriptrunner and arg length in the request
                    for idx in range(0, min((alwayson_script.args_to - alwayson_script.args_from), len(request.alwayson_scripts[alwayson_script_name]["args"]))):
                        script_args[alwayson_script.args_from + idx] = request.alwayson_scripts[alwayson_script_name]["args"][idx]
        return script_args

    def text2imgapi(self, txt2imgreq: models.StableDiffusionTxt2ImgProcessingAPI):
        script_runner = scripts.scripts_txt2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(False)
            ui.create_ui()
        if not self.default_script_arg_txt2img:
            self.default_script_arg_txt2img = self.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = self.get_selectable_script(txt2imgreq.script_name, script_runner)

        populate = txt2imgreq.copy(update={  # Override __init__ params
            "sampler_name": validate_sampler_name(txt2imgreq.sampler_name or txt2imgreq.sampler_index),
            "do_not_save_samples": not txt2imgreq.save_images,
            "do_not_save_grid": not txt2imgreq.save_images,
        })
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        args = vars(populate)
        args.pop('script_name', None)
        args.pop('script_args', None) # will refeed them to the pipeline directly after initializing them
        args.pop('alwayson_scripts', None)

        script_args = self.init_script_args(txt2imgreq, self.default_script_arg_txt2img, selectable_scripts, selectable_script_idx, script_runner)

        send_images = args.pop('send_images', True)
        args.pop('save_images', None)

        with self.queue_lock:
            with closing(StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)) as p:
                p.is_api = True
                p.scripts = script_runner
                p.outpath_grids = opts.outdir_txt2img_grids
                p.outpath_samples = opts.outdir_txt2img_samples

                try:
                    shared.state.begin(job="scripts_txt2img")
                    if selectable_scripts is not None:
                        p.script_args = script_args
                        processed = scripts.scripts_txt2img.run(p, *p.script_args) # Need to pass args as list here
                    else:
                        p.script_args = tuple(script_args) # Need to pass args as tuple here
                        processed = process_images(p)
                finally:
                    shared.state.end()
                    shared.total_tqdm.clear()

        b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []

        return models.TextToImageResponse(images=b64images, parameters=vars(txt2imgreq), info=processed.js())

    def img2imgapi(self, img2imgreq: models.StableDiffusionImg2ImgProcessingAPI):
        init_images = img2imgreq.init_images
        if init_images is None:
            raise HTTPException(status_code=404, detail="Init image not found")

        mask = img2imgreq.mask
        if mask:
            mask = mask_decode_to_image(mask)

        script_runner = scripts.scripts_img2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(True)
            ui.create_ui()
        if not self.default_script_arg_img2img:
            self.default_script_arg_img2img = self.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = self.get_selectable_script(img2imgreq.script_name, script_runner)

        populate = img2imgreq.copy(update={  # Override __init__ params
            "sampler_name": validate_sampler_name(img2imgreq.sampler_name or img2imgreq.sampler_index),
            "do_not_save_samples": not img2imgreq.save_images,
            "do_not_save_grid": not img2imgreq.save_images,
            "mask": mask,
        })
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on

        args = vars(populate)
        args.pop('include_init_images', None)  # this is meant to be done by "exclude": True in model, but it's for a reason that I cannot determine.
        args.pop('script_name', None)
        args.pop('script_args', None)  # will refeed them to the pipeline directly after initializing them
        args.pop('alwayson_scripts', None)

        script_args = self.init_script_args(img2imgreq, self.default_script_arg_img2img, selectable_scripts, selectable_script_idx, script_runner)

        send_images = args.pop('send_images', True)
        args.pop('save_images', None)

        with self.queue_lock:
            with closing(StableDiffusionProcessingImg2Img(sd_model=shared.sd_model, **args)) as p:
                p.init_images = [decode_to_image(x) for x in init_images]
                p.is_api = True
                p.scripts = script_runner
                p.outpath_grids = opts.outdir_img2img_grids
                p.outpath_samples = opts.outdir_img2img_samples

                try:
                    shared.state.begin(job="scripts_img2img")
                    if selectable_scripts is not None:
                        p.script_args = script_args
                        processed = scripts.scripts_img2img.run(p, *p.script_args) # Need to pass args as list here
                    else:
                        p.script_args = tuple(script_args) # Need to pass args as tuple here
                        processed = process_images(p)
                finally:
                    shared.state.end()
                    shared.total_tqdm.clear()

        b64images = list(map(encode_pil_to_base64, processed.images)) if send_images else []

        if not img2imgreq.include_init_images:
            img2imgreq.init_images = None
            img2imgreq.mask = None

        return models.ImageToImageResponse(images=b64images, parameters=vars(img2imgreq), info=processed.js())

    def extras_single_image_api(self, req: models.ExtrasSingleImageRequest):
        reqDict = setUpscalers(req)

        reqDict['image'] = decode_to_image(reqDict['image'])

        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=0, image_folder="", input_dir="", output_dir="", save_output=False, **reqDict)

        return models.ExtrasSingleImageResponse(image=encode_pil_to_base64(result[0][0]), html_info=result[1])

    def extras_batch_images_api(self, req: models.ExtrasBatchImagesRequest):
        reqDict = setUpscalers(req)

        image_list = reqDict.pop('imageList', [])
        image_folder = [decode_to_image(x.data) for x in image_list]

        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=1, image_folder=image_folder, image="", input_dir="", output_dir="", save_output=False, **reqDict)

        return models.ExtrasBatchImagesResponse(images=list(map(encode_pil_to_base64, result[0])), html_info=result[1])

    def pnginfoapi(self, req: models.PNGInfoRequest):
        if(not req.image.strip()):
            return models.PNGInfoResponse(info="")

        image = decode_to_image(req.image.strip())
        if image is None:
            return models.PNGInfoResponse(info="")

        geninfo, items = images.read_info_from_image(image)
        if geninfo is None:
            geninfo = ""

        items = {**{'parameters': geninfo}, **items}

        return models.PNGInfoResponse(info=geninfo, items=items)

    def progressapi(self, req: models.ProgressRequest = Depends()):
        # copy from check_progress_call of ui.py

        if shared.state.job_count == 0:
            return models.ProgressResponse(progress=0, eta_relative=0, state=shared.state.dict(), textinfo=shared.state.textinfo)

        # avoid dividing zero
        progress = 0.01

        if shared.state.job_count > 0:
            progress += shared.state.job_no / shared.state.job_count
        if shared.state.sampling_steps > 0:
            progress += 1 / shared.state.job_count * shared.state.sampling_step / shared.state.sampling_steps

        time_since_start = time.time() - shared.state.time_start
        eta = (time_since_start/progress)
        eta_relative = eta-time_since_start

        progress = min(progress, 1)

        shared.state.set_current_image()

        current_image = None
        if shared.state.current_image and not req.skip_current_image:
            current_image = encode_pil_to_base64(shared.state.current_image)

        return models.ProgressResponse(progress=progress, eta_relative=eta_relative, state=shared.state.dict(), current_image=current_image, textinfo=shared.state.textinfo)

    def interrogateapi(self, interrogatereq: models.InterrogateRequest):
        image_b64 = interrogatereq.image
        if image_b64 is None:
            raise HTTPException(status_code=404, detail="Image not found")

        img = decode_to_image(image_b64)
        img = img.convert('RGB')

        # Override object param
        with self.queue_lock:
            if interrogatereq.model == "clip":
                processed = shared.interrogator.interrogate(img)
            elif interrogatereq.model == "deepdanbooru":
                processed = deepbooru.model.tag(img)
            else:
                raise HTTPException(status_code=404, detail="Model not found")

        return models.InterrogateResponse(caption=processed)

    def interruptapi(self):
        shared.state.interrupt()

        return {}

    def unloadapi(self):
        unload_model_weights()

        return {}

    def reloadapi(self):
        reload_model_weights()

        return {}

    def skip(self):
        shared.state.skip()

    def get_config(self):
        options = {}
        for key in shared.opts.data.keys():
            metadata = shared.opts.data_labels.get(key)
            if(metadata is not None):
                options.update({key: shared.opts.data.get(key, shared.opts.data_labels.get(key).default)})
            else:
                options.update({key: shared.opts.data.get(key, None)})

        return options

    def get_all_config(self):
        return shared.opts.data

    def set_config(self, req: Dict[str, Any]):
        checkpoint_name = req.get("sd_model_checkpoint", None)
        if checkpoint_name is not None and checkpoint_name not in checkpoint_aliases:
            raise RuntimeError(f"model {checkpoint_name!r} not found")

        for k, v in req.items():
            shared.opts.set(k, v, is_api=True)

        shared.opts.save(shared.config_filename)
        return

    def get_cmd_flags(self):
        return vars(shared.cmd_opts)

    def get_samplers(self):
        return [{"name": sampler[0], "aliases":sampler[2], "options":sampler[3]} for sampler in sd_samplers.all_samplers]

    def get_upscalers(self):
        return [
            {
                "name": upscaler.name,
                "model_name": upscaler.scaler.model_name,
                "model_path": upscaler.data_path,
                "model_url": None,
                "scale": upscaler.scale,
            }
            for upscaler in shared.sd_upscalers
        ]

    def get_latent_upscale_modes(self):
        return [
            {
                "name": upscale_mode,
            }
            for upscale_mode in [*(shared.latent_upscale_modes or {})]
        ]

    def get_sd_models(self):
        import modules.sd_models as sd_models
        return [{"title": x.title, "model_name": x.model_name, "hash": x.shorthash, "sha256": x.sha256, "filename": x.filename, "config": find_checkpoint_config_near_filename(x)} for x in sd_models.checkpoints_list.values()]

    def get_sd_vaes(self):
        import modules.sd_vae as sd_vae
        return [{"model_name": x, "filename": sd_vae.vae_dict[x]} for x in sd_vae.vae_dict.keys()]

    def get_hypernetworks(self):
        return [{"name": name, "path": shared.hypernetworks[name]} for name in shared.hypernetworks]

    def get_face_restorers(self):
        return [{"name":x.name(), "cmd_dir": getattr(x, "cmd_dir", None)} for x in shared.face_restorers]

    def get_realesrgan_models(self):
        return [{"name":x.name,"path":x.data_path, "scale":x.scale} for x in get_realesrgan_models(None)]

    def get_prompt_styles(self):
        styleList = []
        for k in shared.prompt_styles.styles:
            style = shared.prompt_styles.styles[k]
            styleList.append({"name":style[0], "prompt": style[1], "negative_prompt": style[2]})

        return styleList

    def get_embeddings(self):
        db = sd_hijack.model_hijack.embedding_db

        def convert_embedding(embedding):
            return {
                "step": embedding.step,
                "sd_checkpoint": embedding.sd_checkpoint,
                "sd_checkpoint_name": embedding.sd_checkpoint_name,
                "shape": embedding.shape,
                "vectors": embedding.vectors,
            }

        def convert_embeddings(embeddings):
            return {embedding.name: convert_embedding(embedding) for embedding in embeddings.values()}

        return {
            "loaded": convert_embeddings(db.word_embeddings),
            "skipped": convert_embeddings(db.skipped_embeddings),
        }

    def refresh_checkpoints(self):
        with self.queue_lock:
            shared.refresh_checkpoints()

    def refresh_vae(self):
        with self.queue_lock:
            shared_items.refresh_vae_list()

    def create_embedding(self, args: dict):
        try:
            shared.state.begin(job="create_embedding")
            filename = create_embedding(**args) # create empty embedding
            sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings() # reload embeddings so new one can be immediately used
            return models.CreateResponse(info=f"create embedding filename: {filename}")
        except AssertionError as e:
            return models.TrainResponse(info=f"create embedding error: {e}")
        finally:
            shared.state.end()


    def create_hypernetwork(self, args: dict):
        try:
            shared.state.begin(job="create_hypernetwork")
            filename = create_hypernetwork(**args) # create empty embedding
            return models.CreateResponse(info=f"create hypernetwork filename: {filename}")
        except AssertionError as e:
            return models.TrainResponse(info=f"create hypernetwork error: {e}")
        finally:
            shared.state.end()

    def preprocess(self, args: dict):
        try:
            shared.state.begin(job="preprocess")
            preprocess(**args) # quick operation unless blip/booru interrogation is enabled
            shared.state.end()
            return models.PreprocessResponse(info='preprocess complete')
        except KeyError as e:
            return models.PreprocessResponse(info=f"preprocess error: invalid token: {e}")
        except Exception as e:
            return models.PreprocessResponse(info=f"preprocess error: {e}")
        finally:
            shared.state.end()

    def train_embedding(self, args: dict):
        try:
            shared.state.begin(job="train_embedding")
            apply_optimizations = shared.opts.training_xattention_optimizations
            error = None
            filename = ''
            if not apply_optimizations:
                sd_hijack.undo_optimizations()
            try:
                embedding, filename = train_embedding(**args) # can take a long time to complete
            except Exception as e:
                error = e
            finally:
                if not apply_optimizations:
                    sd_hijack.apply_optimizations()
            return models.TrainResponse(info=f"train embedding complete: filename: {filename} error: {error}")
        except Exception as msg:
            return models.TrainResponse(info=f"train embedding error: {msg}")
        finally:
            shared.state.end()

    def train_hypernetwork(self, args: dict):
        try:
            shared.state.begin(job="train_hypernetwork")
            shared.loaded_hypernetworks = []
            apply_optimizations = shared.opts.training_xattention_optimizations
            error = None
            filename = ''
            if not apply_optimizations:
                sd_hijack.undo_optimizations()
            try:
                hypernetwork, filename = train_hypernetwork(**args)
            except Exception as e:
                error = e
            finally:
                shared.sd_model.cond_stage_model.to(devices.device)
                shared.sd_model.first_stage_model.to(devices.device)
                if not apply_optimizations:
                    sd_hijack.apply_optimizations()
                shared.state.end()
            return models.TrainResponse(info=f"train embedding complete: filename: {filename} error: {error}")
        except Exception as exc:
            return models.TrainResponse(info=f"train embedding error: {exc}")
        finally:
            shared.state.end()

    def get_memory(self):
        try:
            import os
            import psutil
            process = psutil.Process(os.getpid())
            res = process.memory_info() # only rss is cross-platform guaranteed so we dont rely on other values
            ram_total = 100 * res.rss / process.memory_percent() # and total memory is calculated as actual value is not cross-platform safe
            ram = { 'free': ram_total - res.rss, 'used': res.rss, 'total': ram_total }
        except Exception as err:
            ram = { 'error': f'{err}' }
        try:
            import torch
            if torch.cuda.is_available():
                s = torch.cuda.mem_get_info()
                system = { 'free': s[0], 'used': s[1] - s[0], 'total': s[1] }
                s = dict(torch.cuda.memory_stats(shared.device))
                allocated = { 'current': s['allocated_bytes.all.current'], 'peak': s['allocated_bytes.all.peak'] }
                reserved = { 'current': s['reserved_bytes.all.current'], 'peak': s['reserved_bytes.all.peak'] }
                active = { 'current': s['active_bytes.all.current'], 'peak': s['active_bytes.all.peak'] }
                inactive = { 'current': s['inactive_split_bytes.all.current'], 'peak': s['inactive_split_bytes.all.peak'] }
                warnings = { 'retries': s['num_alloc_retries'], 'oom': s['num_ooms'] }
                cuda = {
                    'system': system,
                    'active': active,
                    'allocated': allocated,
                    'reserved': reserved,
                    'inactive': inactive,
                    'events': warnings,
                }
            else:
                cuda = {'error': 'unavailable'}
        except Exception as err:
            cuda = {'error': f'{err}'}
        return models.MemoryResponse(ram=ram, cuda=cuda)

    def launch(self, server_name, port, root_path):
        self.app.include_router(self.router)
        uvicorn.run(self.app, host=server_name, port=port, timeout_keep_alive=shared.cmd_opts.timeout_keep_alive, root_path=root_path)

    def kill_webui(self):
        restart.stop_program()

    def restart_webui(self):
        if restart.is_restartable():
            restart.restart_program()
        return Response(status_code=501)

    def stop_webui(request):
        shared.state.server_command = "stop"
        return Response("Stopping.")

    def post_invocations(self, b64images, quality):
        if shared.generated_images_s3uri:
            bucket, key = shared.get_bucket_and_key(shared.generated_images_s3uri)
            if key.endswith('/'):
                key = key[ : -1]
            images = []
            for b64image in b64images:
                # bytes_data = export_pil_to_bytes(decode_to_image(b64image), quality)
                image_id = datetime.datetime.now().strftime(f"%Y%m%d%H%M%S-{uuid.uuid4()}")
                suffix = opts.samples_format.lower()
                bytes_data = export_pil_to_bytes(decode_to_image(b64image), quality, image_id=image_id)

                print(f"log@{datetime.datetime.now().strftime(f'%Y%m%d%H%M%S')} GenID@{user_input_data['generation_id']} image_id@{image_id}")

                shared.s3_client.put_object(
                    Body=bytes_data,
                    Bucket=bucket,
                    Key=f'{key}/{image_id}.{suffix}'
                )
                images.append(f's3://{bucket}/{key}/{image_id}.{suffix}')
            return images
        else:
            return b64images

    def truncate_content(self, value, limit=1000):
        if isinstance(value, str):  # Only truncate if the value is a string
            if len(value) > limit:
                return value[:limit] + '...'
        return value

    def req_logging(self, obj, indent=1):
        if "__dict__" in dir(obj):  # if value is an object, dive into it
            items = obj.__dict__.items()
        elif isinstance(obj, dict):  # if value is a dictionary, get items
            items = obj.items()
        elif isinstance(obj, list):  # if value is a list, enumerate items
            items = enumerate(obj)
        else:  # if value is not an object or dict or list, just print it
            print("  " * indent + f"{self.truncate_content(obj)}")
            return

        for attr, value in items:
            if value is None or value == {} or value == []:
                continue
            if isinstance(value, (list, dict)) or "__dict__" in dir(value):
                print("  " * indent + f"{attr}:")
                self.req_logging(value, indent + 1)
            else:
                print("  " * indent + f"{attr}: {self.truncate_content(value)}")

    def invocations(self, req: models.InvocationsRequest):
        with self.invocations_lock:
            print("\n")
            print("\n")
            print("\n")
            print(f"\n ----------------------------invocation log@{datetime.datetime.now().strftime(f'%Y%m%d%H%M%S')} --------------------------- ")
            try:
               print("")
               self.req_logging(req)
            except Exception as e:
               print("console Log ran into issue: ", e)
            print(f"log@{datetime.datetime.now().strftime(f'%Y%m%d%H%M%S')} req in invocations: {req}")
            global user_input_data
            user_input_data = {}
            # if 'alwayson_scripts' in req:
            #     if "user_input" in req.alwayson_scripts:
            #         user_input_data = req.alwayson_scripts["user_input"]
            #         req.alwayson_scripts.pop("user_input")
            if req.user_input != None:
                user_input_data = req.user_input
                print(f"\n -----<<<<<<<  UserID@{user_input_data['user_id']}  >>>>>>>----- ")
                print(f"\n -----<<<<<<<  GenID@{user_input_data['generation_id']}  >>>>>>>----- ")
                print(f"\n -----<<<<<<<  WFID@{user_input_data['workflow']}  >>>>>>>----- ")
                print_text = (f"\n Received user_input_data:" +
                              f"user_id={user_input_data['user_id']}," +
                              f"date_taken={user_input_data['date_taken']}," +
                              f"generation_id={user_input_data['generation_id']}," +
                              f"workflow={user_input_data['workflow']}")

                if 'project_id' in user_input_data:
                    print_text = print_text + f",project_id={user_input_data['project_id']},"
                if 'design_library_style' in user_input_data:
                    print_text = print_text + f",design_library_style={user_input_data['design_library_style']},"
                if 'camera' in user_input_data:
                    print_text = print_text + f",camera={user_input_data['camera']},"
                if 'fidelity_level' in user_input_data:
                    print_text = print_text + f",fidelity_level={user_input_data['fidelity_level']},"
                if 'additional_prompt' in user_input_data:
                    print_text = print_text + f",additional_prompt={user_input_data['additional_prompt']},"
                if 'atmosphere' in user_input_data:
                    print_text = print_text + f",atmosphere={user_input_data['atmosphere']},"
                if 'orientation' in user_input_data:
                    print_text = print_text + f",orientation={user_input_data['orientation']},"
                if 'imageRatio' in user_input_data:
                    print_text = print_text + f",imageRatio={user_input_data['imageRatio']}"

                print(print_text)
                # print(f"log@{datetime.datetime.now().strftime(f'%Y%m%d%H%M%S')} user_input processed in invocations")
                # req.pop('user_input', None)
            else:
                print(f"\n !!!!!ERROR user_input_data missing!!!!!")

        try:
            if req.vae != None:
                shared.opts.data['sd_vae'] = req.vae
                refresh_vae_list()

            if req.model != None:
                sd_model_checkpoint = shared.opts.sd_model_checkpoint
                shared.opts.sd_model_checkpoint = req.model
                with self.queue_lock:
                    reload_model_weights()
                if sd_model_checkpoint == shared.opts.sd_model_checkpoint:
                    reload_vae_weights()

            quality = req.quality

            embeddings_s3uri = shared.cmd_opts.embeddings_s3uri
            hypernetwork_s3uri = shared.cmd_opts.hypernetwork_s3uri

            if hypernetwork_s3uri !='':
                shared.s3_download(hypernetwork_s3uri, shared.cmd_opts.hypernetwork_dir)
                shared.reload_hypernetworks()

            if req.options != None:
                options = json.loads(req.options)
                for key in options:
                    shared.opts.data[key] = options[key]

            if req.task == 'text-to-image':
                if embeddings_s3uri != '':
                    response = requests.get('http://0.0.0.0:8080/controlnet/model_list', params={'update': True})
                    print('Controlnet models: ', response.text)

                    shared.s3_download(embeddings_s3uri, shared.cmd_opts.embeddings_dir)
                    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
                response = self.text2imgapi(req.txt2img_payload)
                response.images = self.post_invocations(response.images, quality)
                response.parameters.clear()
                oldinfo = json.loads(response.info)
                if "all_prompts" in oldinfo:
                    oldinfo.pop("all_prompts", None)
                if "all_negative_prompts" in oldinfo:
                    oldinfo.pop("all_negative_prompts", None)
                if "infotexts" in oldinfo:
                    oldinfo.pop("infotexts", None)
                response.info = json.dumps(oldinfo)
                return response
            elif req.task == 'image-to-image':
                response = requests.get('http://0.0.0.0:8080/controlnet/model_list', params={'update': True})
                print('Controlnet models: ', response.text)
                # response = requests.get('http://0.0.0.0:8080/sam/heartbeat')
                # print(f'\nsam/heartbeat: {response.text}\n')
                # response = requests.get('http://0.0.0.0:8080/sam/sam-model')
                # print(f'\nsam/sam-model: {response.text}\n')
                #
                #
                # if 'user_input_data' in globals():
                #     # global user_input_data
                #     if user_input_data['workflow'] in ["style", "image"]:
                #         print(f"In {user_input_data['workflow']}")

                if embeddings_s3uri != '':
                    shared.s3_download(embeddings_s3uri, shared.cmd_opts.embeddings_dir)
                    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
                response = self.img2imgapi(req.img2img_payload)
                response.images = self.post_invocations(response.images, quality)
                response.parameters.clear()
                oldinfo = json.loads(response.info)
                if "all_prompts" in oldinfo:
                    oldinfo.pop("all_prompts", None)
                if "all_negative_prompts" in oldinfo:
                    oldinfo.pop("all_negative_prompts", None)
                if "infotexts" in oldinfo:
                    oldinfo.pop("infotexts", None)
                response.info = json.dumps(oldinfo)
                return response
            elif req.task == 'upscale_from_feed':
                # only get the one image (in base64)
                intermediate_image = self.img2imgapi(req.img2img_payload).images
                # print('finished intermediate img2img')
                try:
                    # update the base64 image # note might need to change to req.extras_single_payload['image'] if this does not work
                    req.extras_single_payload.image = intermediate_image[0]
                    response = self.extras_single_image_api(req.extras_single_payload)
                    response.image = self.post_invocations([response.image], quality)[0]
                    response.parameters.clear()
                    oldinfo = json.loads(response.info)
                    if "all_prompts" in oldinfo:
                        oldinfo.pop("all_prompts", None)
                    if "all_negative_prompts" in oldinfo:
                        oldinfo.pop("all_negative_prompts", None)
                    if "infotexts" in oldinfo:
                        oldinfo.pop("infotexts", None)
                    response.info = json.dumps(oldinfo)
                    # print(f"log@{datetime.datetime.now().strftime(f'%Y%m%d%H%M%S')} ### get_cmd_flags is {self.get_cmd_flags()}")
                    return response
                except Exception as e:  # this is in fact obselete, because there will be a earlier return if OOM, won't reach here, but leaving here just in case
                    print(
                        f"An error occurred: {e}, step one upscale failed, reverting to just 4x upscale without Img2Img process")
            elif req.task == 'extras-single-image':
                response = self.extras_single_image_api(req.extras_single_payload)
                response.image = self.post_invocations([response.image], quality)[0]
                if "info" in response:
                    oldinfo = json.loads(response.info)
                    if "all_prompts" in oldinfo:
                        oldinfo.pop("all_prompts", None)
                    if "all_negative_prompts" in oldinfo:
                        oldinfo.pop("all_negative_prompts", None)
                    if "infotexts" in oldinfo:
                        oldinfo.pop("infotexts", None)
                    response.info = json.dumps(oldinfo)
                return response
            elif req.task == 'extras-batch-images':
                response = self.extras_batch_images_api(req.extras_batch_payload)
                response.images = self.post_invocations(response.images, quality)
                return response
            elif req.task == 'interrogate':
                response = self.interrogateapi(req.interrogate_payload)
                return response

            elif req.task == 'get-progress':
                response = self.progressapi(req.progress_payload)
                print(response)
                return response
            elif req.task == 'get-options':
                response = self.get_config()
                return response
            elif req.task == 'get-SDmodels':
                response = self.get_sd_models()
                return response
            elif req.task == 'get-upscalers':
                response = self.get_upscalers()
                return response
            elif req.task == 'get-memory':
                response = self.get_memory()
                return response
            elif req.task == 'get-cmd-flags':
                response = self.get_cmd_flags()
                return response
            elif req.task == 'do-nothing':
                print("nothing has happened")
                return "nothing has happened"

            elif req.task.startswith('/'):
                if req.extra_payload:
                    response = requests.post(url=f'http://0.0.0.0:8080{req.task}', json=req.extra_payload)
                else:
                    response = requests.get(url=f'http://0.0.0.0:8080{req.task}')
                if response.status_code == 200:
                    return json.loads(response.text)
                else:
                    raise HTTPException(status_code=response.status_code, detail=response.text)
            else:
                return models.InvocationsErrorResponse(error = f'Invalid task - {req.task}')

        except Exception as e:
            traceback.print_exc()
            return models.InvocationsErrorResponse(error = str(e))

    def ping(self):
        return {'status': 'Healthy'}
