from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render

from stream.streaming import stream
from stream.stt import stt_mp4


def index(request):
    return render(request, 'index.html')


def video_feed1(request):
    source = 'videos/traffic_25fps.mp4'
    # source = "rtsp://localhost:8554/stream"
    return StreamingHttpResponse(stream(source), content_type='multipart/x-mixed-replace; boundary=frame')


def stt(request):
    if request.method == "POST":
        audio_file = request.FILES.get('audio')
        text = stt_mp4(audio_file)

        return JsonResponse({
            "success" : 200,
            "text" : text
        })