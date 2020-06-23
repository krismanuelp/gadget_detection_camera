import argparse

from picamera import PiCamera
from time import time, strftime
import time

from aiy.leds import Leds
from aiy.leds import PrivacyLed
from aiy.toneplayer import TonePlayer

from aiy.vision.inference import CameraInference
import aiy_gadget_detection

# Sound setup
MODEL_LOAD_SOUND = ('C6w', 'c6w', 'C6w')
BEEP_SOUND = ('E6q', 'C6q')
player = TonePlayer(gpio=22, bpm=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_frames',
        '-f',
        type=int,
        dest='num_frames',
        default=-1,
        help='Sets the number of frames to run for, otherwise runs forever.')

    parser.add_argument(
        '--num_pics',
        '-p',
        type=int,
        dest='num_pics',
        default=-1,
        help='Sets the max number of pictures to take, otherwise runs forever.')

    args = parser.parse_args()

    with PiCamera() as camera, PrivacyLed(Leds()):
        camera.sensor_mode = 5
        camera.resolution = (1640, 922)
        camera.start_preview(fullscreen=True)

        with CameraInference(aiy_gadget_detection.model()) as inference:
            print("Camera inference started")
            player.play(*MODEL_LOAD_SOUND)

            last_time = time()
            pics = 0
            save_pic = False

            for f, result in enumerate(inference.run()):

                for i, obj in enumerate(aiy_gadget_detection.get_objects(result, 0.3)):

                    print('%s Object #%d: %s' % (strftime("%Y-%m-%d-%H:%M:%S"), i, str(obj)))
                    # x, y, width, height = obj.bounding_box
                    if obj.label == 'HP' or 'Laptop' or 'Tablet':
                        save_pic = True
                        sleep(5)
                        player.play(*BEEP_SOUND)

                # menyimpan gambar jika lebih dari 1 gawai terdeteksi
                if save_pic:
                    # menyimpan clean image (set true jika ingin)
                    camera.capture("images/image_%s.jpg" % strftime("%Y%m%d-%H%M%S"))
                    pics += 1
                    save_pic = False

                if f == args.num_frames or pics == args.num_pics:
                    break

                now = time()
                duration = (now - last_time)

                if duration > 0.50:
                    print("Total process time: %s seconds. Bonnet inference time: %s ms " %
                          (duration, result.duration_ms))

                last_time = now

        camera.stop_preview()


if __name__ == '__main__':
    main()
