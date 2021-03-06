import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid
import numpy as np
import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from tflite_class import TfLiteModel

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mask_model_path = "mask_model.tflite"
a_g_model_path = "age_gender_model.tflite"

mask_model = TfLiteModel(mask_model_path)
a_g_model = TfLiteModel(a_g_model_path)


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform

    async def recv(self):
        frame = await self.track.recv()

        if self.transform == "Mask-detection":
            # define , load models
            img = frame.to_ndarray(format="bgr24")
            faces = face_detect.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces:
                # face images processing
                face_img = img[y:y + h, x:x + w]
                face_img1 = input_process(face_img)

                # predict mask / no mask

                mask_pred = mask_model.model_predict(face_img1)

                if mask_pred > 0:  # no mask
                    color = (0, 0, 255)
                    text = 'No Mask'
                else:
                    text = 'MASK'
                    color = (0, 255, 0)
                cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "Age-Gender-detect":
            # perform edge detection
            img = frame.to_ndarray(format="bgr24")
            faces = face_detect.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces:
                # face images processing
                face_img = img[y:y + h, x:x + w]
                face_img2 = input_process(face_img, shape=(64, 64))
                gender_pred, age_pred = a_g_model.model_predict(face_img2)
                if gender_pred[0][0] > 0.5:
                    gender = 'Female'
                else:
                    gender = 'Male'
                ages = np.arange(0, 101).reshape(101, 1)
                age_pred = age_pred.dot(ages).flatten()
                color = (0, 0, 255)
                text = gender + '  ' + str(int(age_pred))
                cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "Detect-all":
            # rotate image
            img = frame.to_ndarray(format="bgr24")
            faces = face_detect.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces:
                # face images processing
                face_img = img[y:y + h, x:x + w]
                face_img1 = input_process(face_img)

                # predict mask / no mask

                mask_pred = mask_model.model_predict(face_img1)

                if mask_pred > 0:  # no mask
                    face_img2 = input_process(face_img, shape=(64, 64))
                    gender_pred, age_pred = a_g_model.model_predict(face_img2)
                    if gender_pred[0][0] > 0.5:
                        gender = 'Female'
                    else:
                        gender = 'Male'
                    ages = np.arange(0, 101).reshape(101, 1)
                    age_pred = age_pred.dot(ages).flatten()
                    mask_pred = 'No Mask'
                    color = (0, 0, 255)
                    text = mask_pred + '  ' + gender + '  ' + str(int(age_pred))
                else:
                    text = 'MASK'
                    color = (0, 255, 0)
                cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        else:
            return frame


def input_process(image, shape=(224, 224)):
    out_image = cv2.resize(image, shape)
    out_image = out_image[np.newaxis]
    out_image = np.array(out_image, dtype=np.float32)
    return out_image


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    # player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    '''if args.write_audio:
        recorder = MediaRecorder(args.write_audio)
    else:
        recorder = MediaBlackhole()
'''

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            local_video = VideoTransformTrack(
                track, transform=params["video_transform"]
            )
            pc.addTrack(local_video)

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            # await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    # await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def create_app():
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post("/offer", offer)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    # parser.add_argument(  # change port salahudddin
    #   "--port", type=int, default=9090, help="Port for HTTP server (default: 8080)"
    # )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--write-audio", help="Write received audio to a file")
    args = parser.parse_args()
    args.verbose = False
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    # app = web.Application()
    # app.on_shutdown.append(on_shutdown)
    # app.router.add_get("/", index)
    # app.router.add_get("/client.js", javascript)
    # app.router.add_post("/offer", offer)
    web.run_app(create_app(), access_log=None, host=args.host, port=8585, ssl_context=ssl_context)
