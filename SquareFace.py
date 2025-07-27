from kivy.app import App
from kivy.uix.screenmanager import Screen, ScreenManager, SlideTransition
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image, AsyncImage
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
import cv2
import os
import pygame
import ffmpeg
from moviepy.audio.io.AudioFileClip import AudioFileClip
from kivy.config import Config
import time
import numpy
import sys
from keras.models import model_from_json
import keras.utils as ima


area_for_scan = ""
selected_file_path = ""
path_to_app = ""
detections_path = path_to_app + "/detections"
path_to_app_assets = "/assets"
model_scale_factor = 1.2
cam_port = 0
current_screen = "ChooseInputScreen"
list_of_emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]
invalid_file_extensions = ["peg", "jpg", "gif", "png", "iff", "psd", "pdf", "eps", "ndd", ".ai", "raw"]


def get_body_classifier():
    if area_for_scan == "face":
        return cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_frontalface_default.xml")
    if area_for_scan == "body":
        return cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_fullbody.xml")
    if area_for_scan == "eye":
        return cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_eye.xml")
    if area_for_scan == "smile":
        return cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_smile.xml")
    if area_for_scan == "text":
        return cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_license_plate_number.xml")
    raise Exception("No body classifier found for " + area_for_scan)


def create_detections_directory_if_not_exists():
    if not os.path.exists(detections_path):
        os.makedirs(detections_path)


class WelcomeScreen(Screen):
    def __init__(self, **kwargs):
        super(WelcomeScreen, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        welcome_label = Label(text="WELCOME TO SQUARE FACE", pos_hint={"x": -0.003, "y": 0.20}, color=(0.309, 0.933, 0.078, 4))
        choose_label = Label(text="CHOOSE WHAT YOU WOULD LIKE TO DETECT", pos_hint={"x": -0.004, "y": 0.123},
                       color=(0.309, 0.933, 0.078, 4))
        note_label = Label(text="NOTE: body and text recognitions require more processing power", pos_hint={"x": -0.004, "y": 0.07},
                             color=(0.309, 0.933, 0.078, 4))
        choose_face_button = Button(text="FACE", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.04, "y": 0.38},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.20, 0.15))
        choose_face_button.bind(on_press=self.switch_to_face_screen)
        choose_body_button = Button(text="BODY", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.28, "y": 0.38},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.20, 0.15))
        choose_body_button.bind(on_press=self.switch_to_body_screen)
        choose_eye_button = Button(text="EYE", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.52, "y": 0.38},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.20, 0.15))
        choose_eye_button.bind(on_press=self.switch_to_eye_screen)
        choose_smile_button = Button(text="SMILE", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.76, "y": 0.38},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.20, 0.15))
        choose_smile_button.bind(on_press=self.switch_to_smile_screen)
        choose_license_button = Button(text="TEXT", background_color=(0.309, 0.933, 0.078, 4),
                                       pos_hint={"x": 0.05, "y": 0.22},
                                       color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.10), font_size="13sp")
        choose_license_button.bind(on_press=self.switch_to_license_screen)
        choose_emotion_button = Button(text="EMOTION", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.05, "y": 0.10},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.10), font_size="13sp")
        choose_emotion_button.bind(on_press=self.switch_to_emotion_screen)
        camera_port_label = Label(text="SPECIFY YOUR CAMERA PORT IF NEEDED (Default is 0)",
                       pos_hint={"x": 0.14, "y": -0.25},
                       color=(0.309, 0.933, 0.078, 4))
        self.cam_port_input = TextInput(text="0",
                                        multiline=False,
                                        size_hint=(0.55, .07),
                                        pos_hint={'x': 0.37, 'y': 0.15},
                                        background_color=(0.309, 0.933, 0.078, 4))
        logo_image = Image(source=path_to_app_assets + "/logo.png", size_hint=(0.15, .3), pos_hint={"x": 0.423, "y": 0.71})

        layout.add_widget(welcome_label)
        layout.add_widget(choose_label)
        layout.add_widget(note_label)
        layout.add_widget(camera_port_label)
        layout.add_widget(choose_face_button)
        layout.add_widget(choose_body_button)
        layout.add_widget(choose_eye_button)
        layout.add_widget(choose_smile_button)
        layout.add_widget(choose_license_button)
        layout.add_widget(choose_emotion_button)
        layout.add_widget(logo_image)
        layout.add_widget(self.cam_port_input)
        self.add_widget(layout)

    def switch_to_face_screen(self, *args):
        global model_scale_factor
        global area_for_scan
        model_scale_factor = 1.2
        area_for_scan = "face"
        self._apply_cam_port()
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "ChooseInputScreen"

    def switch_to_body_screen(self, *args):
        global model_scale_factor
        global area_for_scan
        # lower scale factor for recognition to be more precise
        model_scale_factor = 1.035
        area_for_scan = "body"
        self._apply_cam_port()
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "ChooseInputScreen"

    def switch_to_eye_screen(self, *args):
        global model_scale_factor
        global area_for_scan
        model_scale_factor = 1.2
        area_for_scan = "eye"
        self._apply_cam_port()
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "ChooseInputScreen"

    def switch_to_smile_screen(self, *args):
        global model_scale_factor
        global area_for_scan
        model_scale_factor = 1.2
        area_for_scan = "smile"
        self._apply_cam_port()
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "ChooseInputScreen"

    def switch_to_license_screen(self, *args):
        global model_scale_factor
        global area_for_scan
        # lower scale factor for recognition to be more precise
        model_scale_factor = 1.06
        area_for_scan = "text"
        self._apply_cam_port()
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "ChooseInputScreen"

    def switch_to_emotion_screen(self, *args):
        global model_scale_factor
        global area_for_scan
        model_scale_factor = 1.2
        area_for_scan = "emotion"
        self._apply_cam_port()
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "ChooseInputScreen"

    def _apply_cam_port(self):
        global cam_port
        try:
            cam_port = int(self.cam_port_input.text)
        except Exception as e:
            print("Error:" + str(e))
            cam_port = 0


class ChooseInputScreen(Screen):
    def __init__(self, **kwargs):
        super(ChooseInputScreen, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))
        self.current_selection_label = Label(text="", pos_hint={"x": -0.003, "y": 0.40},
                                  color=(0.309, 0.933, 0.078, 4))
        choose_type_label = Label(text="CHOOSE TYPE OF FILE", pos_hint={"x": -0.003, "y": 0.27}, color=(0.309, 0.933, 0.078, 4))
        note_label = Label(text="NOTE: make sure it is not too bright and not too dark for the recognition to work presicely",
                           pos_hint={"x": -0.003, "y": 0.2}, color=(0.309, 0.933, 0.078, 4))
        choose_video_button = Button(text="VIDEO", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.20, "y": 0.15},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20))
        choose_video_button.bind(on_press=self.switch_to_video_selection_screen)
        choose_video_from_camera_button = Button(text="VIDEO FROM YOUR CAMERA", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.15},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12sp")
        choose_video_from_camera_button.bind(on_press=self.switch_to_cam_video_reading_screen)
        choose_image_button = Button(text="IMAGE", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.20, "y": 0.45},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20))
        choose_image_button.bind(on_press=self.switch_to_image_selection_screen)
        choose_image_from_camera_button = Button(text="IMAGE FROM YOUR CAMERA", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.45},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12sp")
        choose_image_from_camera_button.bind(on_press=self.switch_to_cam_image_reading_screen)
        go_back_button = Button(text="< BACK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.035, "y": 0.85},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.2, 0.1))
        go_back_button.bind(on_press=self.switch_to_welcome_screen)

        layout.add_widget(self.current_selection_label)
        layout.add_widget(choose_type_label)
        layout.add_widget(note_label)
        layout.add_widget(choose_video_button)
        layout.add_widget(choose_video_from_camera_button)
        layout.add_widget(choose_image_button)
        layout.add_widget(choose_image_from_camera_button)
        layout.add_widget(go_back_button)
        self.add_widget(layout)

    def on_enter(self, *args):
        self.current_selection_label.text = "CURRENT SELECTION: " + area_for_scan.upper()

    def switch_to_welcome_screen(self, *args):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = "WelcomeScreen"

    def switch_to_image_selection_screen(self, *args):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "ChooseImageScreen"

    def switch_to_cam_image_reading_screen(self, *args):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "ChooseImageCamInputScreen"

    def switch_to_video_selection_screen(self, *args):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "ChooseVideoScreen"

    def switch_to_cam_video_reading_screen(self, *args):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "ChooseVideoCamInputScreen"


class ChooseImageScreen(Screen):
    def __init__(self, **kwargs):
        super(ChooseImageScreen, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))
        self.popup_layout = FloatLayout(size=(175, 300))

        self.current_selection_label = Label(text="", pos_hint={"x": -0.003, "y": 0.40},
                                             color=(0.309, 0.933, 0.078, 4))
        title_label = Label(text="IMAGE",
                       pos_hint={"x": 0.194, "y": 0.78},
                       color=(0.309, 0.933, 0.078, 4), size_hint=(0.62, .07))
        path_to_image_label = Label(text="WRITE FULL PATH TO AN IMAGE (i.e. /Users/joe/img.png) OR PRESS FILE ICON TO CHOOSE FILE",
                       pos_hint={"x": 0.01, "y": 0.24},
                       color=(0.309, 0.933, 0.078, 4))
        self.path_to_image_input = TextInput(hint_text='PATH TO AN IMAGE',
                                             multiline=False,
                                             size_hint=(0.55, .07),
                                             pos_hint={'x': 0.2, 'y': 0.57},
                                             background_color=(0.309, 0.933, 0.078, 4))
        save_image_button = Button(text="SAVE FULL IMAGE >", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.33},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12.5sp")
        save_image_button.bind(on_press=self.scan_and_save_result)
        save_image_cropped_result_button = Button(text="SAVE ONLY DETECTED AREAS >",
                                                  background_color=(0.309, 0.933, 0.078, 4),
                                                  pos_hint={"x": 0.53, "y": 0.08},
                                                  color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20),
                                                  font_size="12.5sp")
        save_image_cropped_result_button.bind(on_press=self.scan_and_save_cropped_result)
        show_image_result_button = Button(text="SHOW RESULT", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.20, "y": 0.33},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12.5sp")
        show_image_result_button.bind(on_press=self.scan_and_show_result)
        show_image_cropped_result_button = Button(text="SHOW ONLY DETECTED AREAS", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.20, "y": 0.08},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12.5sp")
        show_image_cropped_result_button.bind(on_press=self.scan_and_show_cropped_result)

        open_file_finder_button = Button(text="", background_normal=path_to_app_assets + "/file.png", background_down=path_to_app_assets + "/file.png", pos_hint={"x": 0.75, "y": 0.56}, size_hint=(0.075, 0.095))
        open_file_finder_button.bind(on_press=self.open_file_chooser)
        self.file_chooser_icon_view = FileChooserIconView(dirselect=True)
        close_cancel_file_finder_button = Button(text="CANCEL", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.005, "y": 1.015},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.15, 0.05))
        close_cancel_file_finder_button.bind(on_press=self.close_file_chooser)
        close_apply_file_finder_button = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4),
                              pos_hint={"x": 0.84, "y": 1.015},
                              color=(0.141, 0.054, 0.078, 4), size_hint=(0.15, 0.05))
        close_apply_file_finder_button.bind(on_press=self.apply_file_selection)

        go_back_button = Button(text="< BACK", background_color=(0.309, 0.933, 0.078, 4),
                                pos_hint={"x": 0.035, "y": 0.85},
                                color=(0.141, 0.054, 0.078, 4), size_hint=(0.2, 0.1))
        go_back_button.bind(on_press=self.switch_to_choose_screen)

        layout.add_widget(self.current_selection_label)
        layout.add_widget(self.path_to_image_input)
        layout.add_widget(path_to_image_label)
        layout.add_widget(title_label)
        layout.add_widget(save_image_button)
        layout.add_widget(go_back_button)
        layout.add_widget(show_image_result_button)
        layout.add_widget(show_image_cropped_result_button)
        layout.add_widget(save_image_cropped_result_button)
        layout.add_widget(open_file_finder_button)
        self.popup_layout.add_widget(self.file_chooser_icon_view)
        self.popup_layout.add_widget(close_cancel_file_finder_button)
        self.popup_layout.add_widget(close_apply_file_finder_button)
        self.add_widget(layout)

        self.select_file_popup = Popup(title="Select your file", content=self.popup_layout)

        self.loaded_model = ""

    def on_enter(self, *args):
        self.current_selection_label.text = "CURRENT SELECTION: " + area_for_scan.upper()
        self.loaded_model = model_from_json(open(path_to_app_assets + "/neuralnet.json", "r").read())
        self.loaded_model.load_weights(path_to_app_assets + "/weights.h5")

    def open_file_chooser(self, *args):
        self.select_file_popup.open()

    def close_file_chooser(self, *args):
        self.select_file_popup.dismiss()

    def apply_file_selection(self, *args):
        if len(self.file_chooser_icon_view.selection) > 0:
            self.path_to_image_input.text = self.file_chooser_icon_view.selection[0]
            self.select_file_popup.dismiss()
            return
        self.path_to_image_input.text = ""
        self.select_file_popup.dismiss()

    def switch_to_choose_screen(self, *args):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = "ChooseInputScreen"

    def scan_and_save_result(self, *args):
        path_to_image = self.path_to_image_input.text
        try:
            image = cv2.imread(path_to_image)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if area_for_scan != "emotion":
                self._scan_and_save_body_part(image, gray_image)
                return
            self._scan_and_save_emotion(image, gray_image)
        except Exception as e:
            print("Error: " + str(e))
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "ErrorImageSelectionScreen"

    def scan_and_save_cropped_result(self, *args):
        path_to_image = self.path_to_image_input.text
        try:
            image = cv2.imread(path_to_image)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if area_for_scan != "emotion":
                self._scan_and_save_body_part_cropped(image, gray_image)
                return
            self._scan_and_save_emotion_cropped(image, gray_image)
        except Exception as e:
            print("Error: " + str(e))
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "ErrorImageSelectionScreen"

    def scan_and_show_result(self, *args):
        app = self.path_to_image_input.text
        try:
            image = cv2.imread(app)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if area_for_scan != "emotion":
                self._scan_and_show_body_part_result(image, gray_image)
                return
            self._scan_and_show_emotion_result(image, gray_image)
        except Exception as e:
            print("Error: " + str(e))
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "ErrorImageSelectionScreen"

    def scan_and_show_cropped_result(self, *args):
        path_to_image = self.path_to_image_input.text
        try:
            image = cv2.imread(path_to_image)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if area_for_scan != "emotion":
                self._scan_and_show_body_part_cropped_result(image, gray_image)
                return
            self._scan_and_show_emotion_cropped_result(image, gray_image)
        except Exception as e:
            print("Error: " + str(e))
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "ErrorImageSelectionScreen"

    def _scan_and_save_body_part(self, image, gray_image):
        body_classifier = get_body_classifier()
        detected_areas = body_classifier.detectMultiScale(gray_image, scaleFactor=model_scale_factor, minNeighbors=5)
        if detected_areas == ():
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "NoDetectionsIdentifiedOnImageScreen"
        else:
            for (x, y, z, w) in detected_areas:
                cv2.rectangle(image, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
            cv2.imwrite(path_to_app + "/detection.png", image)
            self.manager.transition = SlideTransition(direction="left")
            self.manager.current = "ProcessedScreen"

    def _scan_and_save_emotion(self, image, gray_image):
        face_classifier = cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_frontalface_default.xml")
        detected_areas = face_classifier.detectMultiScale(image, scaleFactor=model_scale_factor, minNeighbors=5)
        for (x, y, z, w) in detected_areas:
            cv2.rectangle(image, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
            cut_gray_image = gray_image[y: y + w, x: x + z]
            resized_cut_gray_image = cv2.resize(cut_gray_image, (48, 48))
            gray_image_array = ima.img_to_array(resized_cut_gray_image)
            gray_image_array_expanded = numpy.expand_dims(gray_image_array, axis=0)
            gray_image_array_expanded /= 255
            prediction = self.loaded_model.predict(gray_image_array_expanded)
            final_prediction = numpy.argmax(prediction[0])
            result = list_of_emotions[final_prediction]
            if z >= 210 and w >= 210:
                cv2.putText(image, result, (int(x + 5), int(y + 45)), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                            (0, 0, 255), 2)
                continue
            cv2.putText(image, result, (int(x + 5), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 0, 255), 1)

        cv2.imwrite(path_to_app + "/detection.png", image)
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "ProcessedScreen"

    def _scan_and_save_body_part_cropped(self, image, gray_image):
        body_classifier = get_body_classifier()
        detection = body_classifier.detectMultiScale(gray_image, scaleFactor=model_scale_factor, minNeighbors=5)

        create_detections_directory_if_not_exists()
        picker = 0
        if detection == ():
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "NoDetectionsIdentifiedOnImageScreen"
        else:
            for (x, y, z, w) in detection:
                picker += 1
                image_cropped = image[y: y + w, x: x + z]
                image_result = cv2.resize(image_cropped, (520, 400))
                cv2.imwrite(detections_path + "/detection" + str(picker) + ".png", image_result)
            self.manager.transition = SlideTransition(direction="left")
            self.manager.current = "ProcessedScreen"

    def _scan_and_save_emotion_cropped(self, image, gray_image):
        face_classifier = cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_frontalface_default.xml")
        detection = face_classifier.detectMultiScale(image, scaleFactor=model_scale_factor, minNeighbors=5)

        if detection == ():
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "NoDetectionsIdentifiedOnImageScreen"
            return

        create_detections_directory_if_not_exists()
        image_count = 0
        for (x, y, z, w) in detection:
            image_count += 1
            image_cropped = image[y: y + w, x: x + z]
            image_cropped_final = cv2.resize(image_cropped, (520, 400))
            cut_gray = gray_image[y: y + w, x: x + z]
            resized_cut_gray = cv2.resize(cut_gray, (48, 48))
            image_array_gray = ima.img_to_array(resized_cut_gray)
            image_array_gray_expanded = numpy.expand_dims(image_array_gray, axis=0)
            image_array_gray_expanded /= 255
            prediction = self.loaded_model.predict(image_array_gray_expanded)
            final_prediction = numpy.argmax(prediction[0])
            result = list_of_emotions[final_prediction]
            cv2.putText(image_cropped_final, result, (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4,
                        (0, 0, 255), 2)
            cv2.imwrite(detections_path + "/detection" + str(image_count) + ".png",
                        image_cropped_final)
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "ProcessedScreen"

    def _scan_and_show_body_part_result(self, image, gray_image):
        body_classifier = get_body_classifier()
        detection = body_classifier.detectMultiScale(gray_image, scaleFactor=model_scale_factor, minNeighbors=5)
        detection_counter = 0

        if detection == ():
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "NoDetectionsIdentifiedOnImageScreen"
            return

        for (x, y, z, w) in detection:
            cv2.rectangle(image, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
            if detection_counter == 0:
                cv2.putText(image, "PRESS SPACE TWICE TO CLOSE THE IMAGE", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (20, 226, 20), 1)
            detection_counter += 1

        cv2.imshow("PREVIEW", image)
        cv2.namedWindow("PREVIEW")
        cv2.waitKey(0)

        retry = True
        while retry:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    retry = False
                    cv2.destroyAllWindows()

    def _scan_and_show_emotion_result(self, image, gray_image):
        face_classifier = cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_frontalface_default.xml")
        detection = face_classifier.detectMultiScale(image, scaleFactor=model_scale_factor, minNeighbors=5)
        detection_counter = 0

        if detection == ():
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "NoDetectionsIdentifiedOnImageScreen"
            return

        for (x, y, z, w) in detection:
            if detection_counter == 0:
                cv2.putText(image, "PRESS SPACE TWICE TO CLOSE THE IMAGE", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (20, 226, 20), 1)
            detection_counter += 1
            cv2.rectangle(image, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
            cut_gray_image = gray_image[y: y + w, x: x + z]
            resized_cut_gray_image = cv2.resize(cut_gray_image, (48, 48))
            gray_image_array = ima.img_to_array(resized_cut_gray_image)
            gray_image_array_expanded = numpy.expand_dims(gray_image_array, axis=0)
            gray_image_array_expanded /= 255
            prediction = self.loaded_model.predict(gray_image_array_expanded)
            final_prediction = numpy.argmax(prediction[0])
            result = list_of_emotions[final_prediction]
            if z >= 210 and w >= 210:
                cv2.putText(image, result, (int(x + 5), int(y + 45)), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                            (0, 0, 255), 2)
                continue
            cv2.putText(image, result, (int(x + 5), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 0, 255), 1)
        cv2.imshow("PREVIEW", image)
        cv2.namedWindow("PREVIEW")
        cv2.waitKey(0)

        retry = True
        while retry:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    retry = False
                    cv2.destroyAllWindows()

    def _scan_and_show_body_part_cropped_result(self, image, gray_image):
        body_classifier = get_body_classifier()
        detection = body_classifier.detectMultiScale(gray_image, scaleFactor=model_scale_factor, minNeighbors=5)

        if detection == ():
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "NoDetectionsIdentifiedOnImageScreen"
            return

        for index, (x, y, z, w) in enumerate(detection):
            image_cropped = image[y: y + w, x: x + z]
            image_cropped_resized = cv2.resize(image_cropped, (520, 400))
            if index == len(detection) - 1:
                cv2.putText(image_cropped_resized, "PRESS SPACE TWICE TO CLOSE THE IMAGE", (10, 11),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.30,
                            (20, 226, 20), 1)
            else:
                cv2.putText(image_cropped_resized, "PRESS SPACE TO GET TO THE NEXT IMAGE", (10, 11),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.30,
                            (20, 226, 20), 1)
            cv2.imshow("PREVIEW", image_cropped_resized)
            cv2.namedWindow("PREVIEW")
            cv2.waitKey(0)

        retry = True
        while retry:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    retry = False
                    cv2.destroyWindow("PREVIEW")

    def _scan_and_show_emotion_cropped_result(self, image, gray_image):
        face_classifier = cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_frontalface_default.xml")
        detection = face_classifier.detectMultiScale(image, scaleFactor=model_scale_factor, minNeighbors=5)

        if detection == ():
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "NoDetectionsIdentifiedOnImageScreen"
            return

        for index, (x, y, z, w) in enumerate(detection):
            image_cropped = image[y: y + w, x: x + z]
            image_cropped_resized = cv2.resize(image_cropped, (520, 400))
            cut_gray_image = gray_image[y: y + w, x: x + z]
            resized_cut_gray_image = cv2.resize(cut_gray_image, (48, 48))
            gray_image_array = ima.img_to_array(resized_cut_gray_image)
            gray_image_array_expanded = numpy.expand_dims(gray_image_array, axis=0)
            gray_image_array_expanded /= 255
            prediction = self.loaded_model.predict(gray_image_array_expanded)
            final_prediction = numpy.argmax(prediction[0])
            result = list_of_emotions[final_prediction]
            cv2.putText(image_cropped_resized, result, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4,
                        (0, 0, 255), 2)
            if index == len(detection) - 1:
                cv2.putText(image_cropped_resized, "PRESS SPACE TWICE TO CLOSE THE IMAGE", (10, 11),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.30,
                            (20, 226, 20), 1)
            else:
                cv2.putText(image_cropped_resized, "PRESS SPACE TO GET TO THE NEXT IMAGE", (10, 11),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.30,
                            (20, 226, 20), 1)
            cv2.imshow("PREVIEW", image_cropped_resized)
            cv2.namedWindow("PREVIEW")
            cv2.waitKey(0)

        retry = True
        while retry:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    retry = False
                    cv2.destroyWindow("PREVIEW")


class ChooseImageCamInputScreen(Screen):
    def __init__(self, **kwargs):
        super(ChooseImageCamInputScreen, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))
        self.current_selection_label = Label(text="", pos_hint={"x": -0.003, "y": 0.40},
                                             color=(0.309, 0.933, 0.078, 4))
        title_label = Label(text="IMAGE FROM YOUR CAMERA", pos_hint={"x": 0.01, "y": 0.25},
                       color=(0.309, 0.933, 0.078, 4))
        take_and_save_button = Button(text="TAKE AND SAVE FULL IMAGE >", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.45},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.23), font_size="13sp")
        take_and_save_button.bind(on_press=self.scan_and_save_result)
        take_and_show_button = Button(text="TAKE AND SHOW RESULT", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.20, "y": 0.45},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.23), font_size="13sp")
        take_and_show_button.bind(on_press=self.scan_and_show_result)
        take_and_show_cropped_button = Button(text="TAKE AND SHOW DETECTED AREAS", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.20, "y": 0.15},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.23), font_size="13sp")
        take_and_show_cropped_button.bind(on_press=self.scan_and_show_cropped_result)
        take_and_save_cropped_button = Button(text="TAKE AND SAVE DETECTED AREAS >", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.15},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.23), font_size="13sp")
        take_and_save_cropped_button.bind(on_press=self.scan_and_save_cropped_result)
        go_back_button = Button(text="< BACK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.035, "y": 0.85},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.2, 0.1))
        go_back_button.bind(on_press=self.switch_to_choose_input_screen)

        layout.add_widget(self.current_selection_label)
        layout.add_widget(title_label)
        layout.add_widget(take_and_save_button)
        layout.add_widget(take_and_show_button)
        layout.add_widget(take_and_show_cropped_button)
        layout.add_widget(take_and_save_cropped_button)
        layout.add_widget(go_back_button)
        self.add_widget(layout)
        self.loaded_model = ""

    def on_enter(self, *args):
        self.current_selection_label.text = "CURRENT SELECTION: " + area_for_scan.upper()
        self.loaded_model = model_from_json(open(path_to_app_assets + "/neuralnet.json", "r").read())
        self.loaded_model.load_weights(path_to_app_assets + "/weights.h5")

    def switch_to_choose_input_screen(self, *args):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = "ChooseInputScreen"

    def scan_and_save_result(self, *args):
        try:
            image = cv2.VideoCapture(cam_port)
            frame = self._get_capture_frame(image)

            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if area_for_scan != "emotion":
                self._scan_and_save_body_part_result(frame, image, gray_image)
                return
            self._scan_and_save_emotion_result(frame, image, gray_image)
        except Exception as e:
            print("Error: " + str(e))
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "ErrorImageCamScreen"

    def scan_and_save_cropped_result(self, *args):
        try:
            image = cv2.VideoCapture(cam_port)
            frame = self._get_capture_frame(image)

            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if area_for_scan != "emotion":
                self._scan_and_save_body_part_cropped_result(frame, image, gray_image)
                return
            self._scan_and_save_emotion_cropped_result(frame, image, gray_image)
        except Exception as e:
            print("Error: " + str(e))
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "ErrorImageCamScreen"

    def scan_and_show_result(self, *args):
        image = cv2.VideoCapture(cam_port)
        frame = self._get_capture_frame(image)

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if area_for_scan != "emotion":
            self._scan_and_show_body_part_result(frame, image, gray_image)
            return
        self._scan_and_show_emotion_result(frame, image, gray_image)

    def scan_and_show_cropped_result(self, *args):
        image = cv2.VideoCapture(cam_port)
        frame = self._get_capture_frame(image)

        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if area_for_scan != "emotion":
            self._scan_and_show_body_part_cropped_result(frame, gray_image)
            return
        self._scan_and_show_emotion_cropped_result(frame, gray_image)

    def _scan_and_save_body_part_result(self, frame, image, gray_image):
        body_classifier = get_body_classifier()
        detection = body_classifier.detectMultiScale(gray_image, scaleFactor=model_scale_factor, minNeighbors=5)

        if detection == ():
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "NoDetectionsIdentifiedOnImageCamScreen"
            return

        for (x, y, z, w) in detection:
            cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
        cv2.imwrite(path_to_app + "/detection.png", frame)
        image.release()
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "ProcessedScreen"

    def _scan_and_save_emotion_result(self, frame, image, gray_image):
        face_classifier = cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_frontalface_default.xml")
        detection = face_classifier.detectMultiScale(frame, scaleFactor=model_scale_factor, minNeighbors=5)

        if detection == ():
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "NoDetectionsIdentifiedOnImageCamScreen"
            return

        for (x, y, z, w) in detection:
            cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
            cut_gray_image = gray_image[y: y + w, x: x + z]
            resized_cut_gray_image = cv2.resize(cut_gray_image, (48, 48))
            gray_image_array = ima.img_to_array(resized_cut_gray_image)
            gray_image_array_expanded = numpy.expand_dims(gray_image_array, axis=0)
            gray_image_array_expanded /= 255
            prediction = self.loaded_model.predict(gray_image_array_expanded)
            final_prediction = numpy.argmax(prediction[0])
            result = list_of_emotions[final_prediction]
            if z >= 210 and w >= 210:
                cv2.putText(frame, result, (int(x + 5), int(y + 45)), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                            (0, 0, 255), 2)
            else:
                cv2.putText(frame, result, (int(x + 5), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (0, 0, 255), 1)
        cv2.imwrite(path_to_app + "/detection.png", frame)
        image.release()
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "ProcessedScreen"

    def _scan_and_save_body_part_cropped_result(self, frame, image, gray_image):
        body_classifier = get_body_classifier()
        detection = body_classifier.detectMultiScale(gray_image, scaleFactor=model_scale_factor, minNeighbors=5)
        image_count = 0

        if detection == ():
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "NoDetectionsIdentifiedOnImageCamScreen"
            return

        create_detections_directory_if_not_exists()
        for (x, y, z, w) in detection:
            image_count += 1
            image_cropped = frame[y: y + w, x: x + z]
            image_result = cv2.resize(image_cropped, (520, 400))
            cv2.imwrite(detections_path + "/detection" + str(image_count) + ".png", image_result)
        image.release()
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "ProcessedScreen"

    def _scan_and_save_emotion_cropped_result(self, frame, image, gray_image):
        face_classifier = cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_frontalface_default.xml")
        detection = face_classifier.detectMultiScale(gray_image, scaleFactor=model_scale_factor, minNeighbors=5)

        if detection == ():
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "NoDetectionsIdentifiedOnImageCamScreen"
            return

        create_detections_directory_if_not_exists()
        image_count = 0
        for (x, y, z, w) in detection:
            image_count += 1
            frame_cropped = frame[y: y + w, x: x + z]
            frame_cropped_final = cv2.resize(frame_cropped, (520, 400))
            gray_image_cropped = gray_image[y: y + w, x: x + z]
            resized_gray_image_cropped = cv2.resize(gray_image_cropped, (48, 48))
            image_array_gray = ima.img_to_array(resized_gray_image_cropped)
            image_array_gray_expanded = numpy.expand_dims(image_array_gray, axis=0)
            image_array_gray_expanded /= 255
            prediction = self.loaded_model.predict(image_array_gray_expanded)
            final_prediction = numpy.argmax(prediction[0])
            result = list_of_emotions[final_prediction]
            cv2.putText(frame_cropped_final, result, (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4,
                        (0, 0, 255), 2)
            cv2.imwrite(detections_path + "/detection" + str(image_count) + ".png",
                        frame_cropped_final)
        image.release()
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "ProcessedScreen"

    def _scan_and_show_body_part_result(self, frame, image, gray_image):
        body_classifier = get_body_classifier()
        detection = body_classifier.detectMultiScale(gray_image, scaleFactor=model_scale_factor, minNeighbors=5)
        image_counter = 0

        if detection == ():
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "NoDetectionsIdentifiedOnImageCamScreen"
            return

        for (x, y, z, w) in detection:
            cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
            if image_counter == 0:
                cv2.putText(frame, "PRESS SPACE TWICE TO CLOSE THE IMAGE", (10, 15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (20, 226, 20), 1)
            image_counter += 1
        cv2.imshow("PREVIEW", frame)
        cv2.namedWindow("PREVIEW")
        cv2.waitKey(0)
        image.release()

        retry = True
        while retry:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    retry = False
                    cv2.destroyAllWindows()

    def _scan_and_show_emotion_result(self, frame, image, gray_image):
        face_classifier = cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_frontalface_default.xml")
        detection = face_classifier.detectMultiScale(frame, scaleFactor=model_scale_factor, minNeighbors=5)
        image_counter = 0

        if detection == ():
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "NoDetectionsIdentifiedOnImageCamScreen"
            return

        for (x, y, z, w) in detection:
            if image_counter == 0:
                cv2.putText(frame, "PRESS SPACE TWICE TO CLOSE THE IMAGE", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (20, 226, 20), 1)
            cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
            image_counter += 1
            gray_image_cropped = gray_image[y: y + w, x: x + z]
            resized_gray_image_cropped = cv2.resize(gray_image_cropped, (48, 48))
            gray_image_array = ima.img_to_array(resized_gray_image_cropped)
            gray_image_array_expanded = numpy.expand_dims(gray_image_array, axis=0)
            gray_image_array_expanded /= 255
            prediction = self.loaded_model.predict(gray_image_array_expanded)
            final_prediction = numpy.argmax(prediction[0])
            result = list_of_emotions[final_prediction]
            if z >= 210 and w >= 210:
                cv2.putText(frame, result, (int(x + 5), int(y + 45)), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                            (0, 0, 255), 2)
            else:
                cv2.putText(frame, result, (int(x + 5), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (0, 0, 255), 1)
        cv2.imshow("PREVIEW", frame)
        cv2.namedWindow("PREVIEW")
        cv2.waitKey(0)
        image.release()

        retry = True
        while retry:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    retry = False
                    cv2.destroyAllWindows()

    def _scan_and_show_body_part_cropped_result(self, frame, gray_image):
        body_classifier = get_body_classifier()
        detection = body_classifier.detectMultiScale(gray_image, scaleFactor=model_scale_factor, minNeighbors=5)

        if detection == ():
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "NoDetectionsIdentifiedOnImageCamScreen"
            return

        for index, (x, y, z, w) in enumerate(detection):
            image_fin = frame[y: y + w, x: x + z]
            image_fin = cv2.resize(image_fin, (520, 400))
            if index == len(detection) - 1:
                cv2.putText(image_fin, "PRESS SPACE TWICE TO CLOSE THE IMAGE", (10, 11),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.30,
                            (20, 226, 20), 1)
            else:
                cv2.putText(image_fin, "PRESS SPACE TO GET TO THE NEXT IMAGE", (10, 11),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.30,
                            (20, 226, 20), 1)
            cv2.imshow("PREVIEW", image_fin)
            cv2.namedWindow("PREVIEW")
            cv2.waitKey(0)

        retry = True
        while retry:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    retry = False
                    cv2.destroyWindow("PREVIEW")

    def _scan_and_show_emotion_cropped_result(self, frame, gray_image):
        face_classifier = cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_frontalface_default.xml")
        detection = face_classifier.detectMultiScale(frame, scaleFactor=model_scale_factor, minNeighbors=5)

        if detection == ():
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "NoDetectionsIdentifiedOnImageCamScreen"
            return

        for index, (x, y, z, w) in enumerate(detection):
            image_cropped = frame[y: y + w, x: x + z]
            image_result = cv2.resize(image_cropped, (520, 400))
            gray_image_cropped = gray_image[y: y + w, x: x + z]
            resized_gray_image_cropped = cv2.resize(gray_image_cropped, (48, 48))
            gray_image_array = ima.img_to_array(resized_gray_image_cropped)
            gray_image_array_expanded = numpy.expand_dims(gray_image_array, axis=0)
            gray_image_array_expanded /= 255
            prediction = self.loaded_model.predict(gray_image_array_expanded)
            final_prediction = numpy.argmax(prediction[0])
            result = list_of_emotions[final_prediction]
            cv2.putText(image_result, result, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.4,
                        (0, 0, 255), 2)
            if index == len(detection) - 1:
                cv2.putText(image_result, "PRESS SPACE TWICE TO CLOSE THE IMAGE", (10, 11),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.30,
                            (20, 226, 20), 1)
            else:
                cv2.putText(image_result, "PRESS SPACE TO GET TO THE NEXT IMAGE", (10, 11),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.30,
                            (20, 226, 20), 1)
            cv2.imshow("PREVIEW", image_result)
            cv2.namedWindow("PREVIEW")
            cv2.waitKey(0)

        retry = True
        while retry:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    retry = False
                    cv2.destroyWindow("PREVIEW")

    @staticmethod
    def _get_capture_frame(image):
        frame = None
        # looping through 5 frames to avoid getting initial black frame
        for _ in range(5):
            k, frame = image.read()
        return frame


class ChooseVideoScreen(Screen):
    def __init__(self, **kwargs):
        super(ChooseVideoScreen, self).__init__(**kwargs)
        self.layout = FloatLayout(size=(350, 600))
        self.popup_layout = FloatLayout(size=(175, 300))

        self.current_selection_label = Label(text="", pos_hint={"x": -0.003, "y": 0.40},
                                             color=(0.309, 0.933, 0.078, 4))
        title_label = Label(text="VIDEO",
                       pos_hint={"x": 0.194, "y": 0.78},
                       color=(0.309, 0.933, 0.078, 4), size_hint=(0.62, .07))
        write_path_label = Label(text="WRITE FULL PATH TO A VIDEO (i.e. /Users/joe/video.mp4) OR PRESS FILE ICON TO CHOOSE FILE",
                       pos_hint={"x": 0.01, "y": 0.24},
                       color=(0.309, 0.933, 0.078, 4))
        self.file_path_input = TextInput(hint_text='PATH TO A VIDEO',
                                         multiline=False,
                                         size_hint=(0.55, .07),
                                         pos_hint={'x': 0.2, 'y': 0.57},
                                         background_color=(0.309, 0.933, 0.078, 4))
        save_video_button = Button(text="SAVE FULL VIDEO >", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.33},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12.5sp")
        save_video_button.bind(on_press=self.scan_and_save_result)
        show_result_button = Button(text="SHOW RESULT", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.20, "y": 0.33},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12.5sp")
        show_result_button.bind(on_press=self.scan_and_show_result)
        show_result_cropped_button = Button(text="SHOW ONLY DETECTED AREAS", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.20, "y": 0.08},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12.5sp")
        show_result_cropped_button.bind(on_press=self.scan_and_show_cropped_result)
        save_result_cropped_button = Button(text="SAVE ONLY DETECTED AREAS >", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.08},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.28, 0.20), font_size="12.5sp")
        save_result_cropped_button.bind(on_press=self.scan_and_save_cropped_result)
        go_back_button = Button(text="< BACK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.035, "y": 0.85},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.2, 0.1))
        go_back_button.bind(on_press=self.switch_to_choose_input_screen)
        file_finder_button = Button(text="", background_normal=path_to_app_assets + "/file.png",
                             background_down=path_to_app_assets + "/file.png", pos_hint={"x": 0.75, "y": 0.56},
                             size_hint=(0.075, 0.095))
        file_finder_button.bind(on_press=self.open_file_finder)
        self.file_finder_icon_view = FileChooserIconView(dirselect=True)
        close_file_finder_cancel_button = Button(text="CANCEL", background_color=(0.309, 0.933, 0.078, 4),
                                 pos_hint={"x": 0.005, "y": 1.015},
                                 color=(0.141, 0.054, 0.078, 4), size_hint=(0.15, 0.05))
        close_file_finder_ok_button = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4),
                              pos_hint={"x": 0.84, "y": 1.015},
                              color=(0.141, 0.054, 0.078, 4), size_hint=(0.15, 0.05))
        close_file_finder_cancel_button.bind(on_press=self.close_file_finder)
        close_file_finder_ok_button.bind(on_press=self.choose_file)
        self.layout.add_widget(self.current_selection_label)
        self.layout.add_widget(title_label)
        self.layout.add_widget(self.file_path_input)
        self.layout.add_widget(write_path_label)
        self.layout.add_widget(save_video_button)
        self.layout.add_widget(go_back_button)
        self.layout.add_widget(show_result_button)
        self.layout.add_widget(show_result_cropped_button)
        self.layout.add_widget(save_result_cropped_button)
        self.layout.add_widget(file_finder_button)
        self.add_widget(self.layout)
        self.popup_layout.add_widget(self.file_finder_icon_view)
        self.popup_layout.add_widget(close_file_finder_cancel_button)
        self.popup_layout.add_widget(close_file_finder_ok_button)

        self.file_finder_popup = Popup(title="select your file", content=self.popup_layout)

        self.loaded_model = ""

    def on_enter(self, *args):
        self.current_selection_label.text = "CURRENT SELECTION: " + area_for_scan.upper()
        self.loaded_model = model_from_json(open(path_to_app_assets + "/neuralnet.json", "r").read())
        self.loaded_model.load_weights(path_to_app_assets + "/weights.h5")

    def open_file_finder(self, *args):
        self.file_finder_popup.open()

    def close_file_finder(self, *args):
        self.file_finder_popup.dismiss()

    def choose_file(self, *args):
        if len(self.file_finder_icon_view.selection) > 0:
            self.file_path_input.text = self.file_finder_icon_view.selection[0]
            self.file_finder_popup.dismiss()
            return
        self.file_path_input.text = ""
        self.file_finder_popup.dismiss()

    def switch_to_choose_input_screen(self, *args):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = "ChooseInputScreen"

    def scan_and_save_result(self, *args):
        global selected_file_path
        selected_file_path = str(self.file_path_input.text)
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "LoadingAndSavingScannedVideoScreen"

    def scan_and_save_cropped_result(self, *args):
        global selected_file_path
        selected_file_path = str(self.file_path_input.text)
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "LoadingAndSavingScannedVideoCroppedScreen"

    def scan_and_show_result(self, *args):
        app = self.file_path_input.text
        try:
            selected_file_extension = str(self.file_path_input.text[-3:])
            if selected_file_extension.lower() not in invalid_file_extensions:

                # check if selected video can be resized
                video_check = cv2.VideoCapture(app)
                _, frame_raw_check = video_check.read()
                cv2.resize(frame_raw_check, (520, 400))
                video_check.release()

                video = cv2.VideoCapture(app)
                height_test = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
                width_test = video.get(cv2.CAP_PROP_FRAME_WIDTH)
                width = str(width_test)
                height = str(height_test)
                width = float(width)
                height = float(height)
                try:
                    if area_for_scan != "emotion":
                        self._scan_and_show_body_part_result(video, width, height)
                    self._scan_and_show_emotion_result(video, width, height)
                except Exception as e:
                    # at the end of the video exception will be thrown so it is only a warning
                    print("Warning: " + str(e))
                    video.release()
                    cv2.destroyAllWindows()
                    return
            else:
                self.manager.transition = SlideTransition(direction="down")
                self.manager.current = "ErrorVideoSelectionScreen"
        except Exception as e:
            print("Error: " + str(e))
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "ErrorVideoSelectionScreen"

    def scan_and_show_cropped_result(self, *args):
        global selected_file_path
        selected_file_path = str(self.file_path_input.text)
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "LoadingAndShowingResultVideoCroppedScreen"

    def _scan_and_show_emotion_result(self, video, width, height):
        retry = True
        while retry:
            _, frame_raw = video.read()
            frame = cv2.resize(frame_raw, (int(width), int(height)))
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_classifier = cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_frontalface_default.xml")
            detection = face_classifier.detectMultiScale(frame, scaleFactor=model_scale_factor, minNeighbors=5)
            cv2.putText(frame, "PRESS SPACE TWICE TO CLOSE THE VIDEO", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (20, 226, 20), 1)

            for (x, y, z, w) in detection:
                cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                gray_image_cropped = gray_image[y: y + w, x: x + z]
                resized_gray_image_cropped = cv2.resize(gray_image_cropped, (48, 48))
                gray_image_array = ima.img_to_array(resized_gray_image_cropped)
                gray_image_array_expanded = numpy.expand_dims(gray_image_array, axis=0)
                gray_image_array_expanded /= 255
                prediction = self.loaded_model.predict(gray_image_array_expanded)
                final_prediction = numpy.argmax(prediction[0])
                result = list_of_emotions[final_prediction]
                if z >= 210 and w >= 210:
                    cv2.putText(frame, result, (int(x + 5), int(y + 45)), cv2.FONT_HERSHEY_SIMPLEX,
                                1.4,
                                (0, 0, 255), 2)
                else:
                    cv2.putText(frame, result, (int(x + 5), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (0, 0, 255), 1)
            cv2.imshow("PREVIEW", frame)
            cv2.namedWindow("PREVIEW")

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    retry = False
                    cv2.destroyAllWindows()

    @staticmethod
    def _scan_and_show_body_part_result(video, width, height):
        retry = True
        while retry:
            body_classifier = get_body_classifier()

            _, frame_raw = video.read()
            frame = cv2.resize(frame_raw, (int(width), int(height)))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detection = body_classifier.detectMultiScale(gray, scaleFactor=model_scale_factor, minNeighbors=5)
            cv2.putText(frame, "PRESS SPACE ONCE OR TWICE TO CLOSE THE VIDEO", (10, 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (20, 226, 20), 1)

            for (x, y, z, w) in detection:
                cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)

            cv2.imshow("PREVIEW", frame)
            cv2.namedWindow("PREVIEW")
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    retry = False
                    cv2.destroyWindow("PREVIEW")


class ChooseVideoCamInputScreen(Screen):
    def __init__(self, **kwargs):
        super(ChooseVideoCamInputScreen, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        self.current_selection_label = Label(text="", pos_hint={"x": -0.003, "y": 0.40},
                                             color=(0.309, 0.933, 0.078, 4))
        title_label = Label(text="VIDEO FROM YOUR CAMERA", pos_hint={"x": 0.01, "y": 0.25},
                       color=(0.309, 0.933, 0.078, 4))
        record_and_save_button = Button(text="RECORD AND SAVE FULL VIDEO >", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.45},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.23), font_size="13sp")
        record_and_save_button.bind(on_press=self.scan_and_save_result)
        reacord_and_show_result_button = Button(text="RECORD AND SHOW RESULT", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.20, "y": 0.45},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.23), font_size="13sp")
        reacord_and_show_result_button.bind(on_press=self.scan_and_show_result)
        record_and_show_cropped_result_button = Button(text="RECORD AND SHOW DETECTED AREAS", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.20, "y": 0.15},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.23), font_size="13sp")
        record_and_show_cropped_result_button.bind(on_press=self.scan_and_show_cropped_result)
        record_and_save_cropped_result_button = Button(text="RECORD AND SAVE DETECTED AREAS >", background_color=(0.309, 0.933, 0.078, 4),
                      pos_hint={"x": 0.53, "y": 0.15},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.23), font_size="13sp")
        record_and_save_cropped_result_button.bind(on_press=self.scan_and_save_cropped_result)
        go_back_button = Button(text="< BACK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.035, "y": 0.85},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.2, 0.1))
        go_back_button.bind(on_press=self.switch_to_choose_input_screen)

        layout.add_widget(self.current_selection_label)
        layout.add_widget(title_label)
        layout.add_widget(record_and_save_button)
        layout.add_widget(go_back_button)
        layout.add_widget(reacord_and_show_result_button)
        layout.add_widget(record_and_show_cropped_result_button)
        layout.add_widget(record_and_save_cropped_result_button)
        self.add_widget(layout)

        self.loaded_model = ""

    def on_enter(self, *args):
        self.current_selection_label.text = "CURRENT SELECTION: " + area_for_scan.upper()
        self.loaded_model = model_from_json(open(path_to_app_assets + "/neuralnet.json", "r").read())
        self.loaded_model.load_weights(path_to_app_assets + "/weights.h5")

    def switch_to_choose_input_screen(self, *args):
        self.manager.transition = SlideTransition(direction="right")
        self.manager.current = "ChooseInputScreen"

    def scan_and_save_result(self, *args):
        try:
            video = cv2.VideoCapture(cam_port)
            height_value = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width_value = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            width = str(width_value)
            height = str(height_value)
            width = float(width)
            height = float(height)
            capture_to_write = cv2.VideoWriter(path_to_app + "/ProcessedVideo.avi", cv2.VideoWriter_fourcc(*"XVID"), 30,
                                      (int(width), int(height)))
            if area_for_scan != "emotion":
                self._scan_and_save_body_part_result(capture_to_write, video, width, height)
                return
            self._scan_and_save_emotion_result(capture_to_write, video, width, height)

        except Exception as e:
            print("Error: " + str(e))
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "ErrorVideoCamScreen"

    def scan_and_save_cropped_result(self, *args):
        try:
            capture_to_write = cv2.VideoWriter(path_to_app + "/ProcessedVideo.avi", cv2.VideoWriter_fourcc(*"XVID"), 30,
                                      (520, 400))
            video = cv2.VideoCapture(cam_port)
            if area_for_scan != "emotion":
                self._scan_and_save_body_part_cropped_result(capture_to_write, video)
                return
            self._scan_and_save_emotion_cropped_result(capture_to_write, video)
        except Exception as e:
            print("Error: " + str(e))
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "ErrorVideoCamScreen"

    def scan_and_show_result(self, *args):
        try:
            video = cv2.VideoCapture(cam_port)
            height_value = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width_value = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            width = str(width_value)
            height = str(height_value)
            width = float(width)
            height = float(height)
            if area_for_scan != "emotion":
                self._scan_and_show_body_part_result(video, width, height)
                return
            self._scan_and_show_emotion_result(video, width, height)
        except Exception as e:
            print("Error: " + str(e))
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "ErrorVideoCamScreen"

    def scan_and_show_cropped_result(self, *args):
        try:
            video = cv2.VideoCapture(cam_port)
            if area_for_scan != "emotion":
                self._scan_and_show_body_part_cropped_result(video)
                return
            self._scan_and_show_emotion_cropped_result(video)
        except Exception as e:
            print("Error: " + str(e))
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "ErrorVideoCamScreen"

    def _scan_and_save_body_part_result(self, capture_to_write, video, width, height):
        create_detections_directory_if_not_exists()
        retry = True
        frame_count = 0
        while retry:
            _, frame_raw = video.read()
            frame = cv2.resize(frame_raw, (int(width), int(height)))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            body_classifier = get_body_classifier()
            detection = body_classifier.detectMultiScale(gray_frame, scaleFactor=model_scale_factor, minNeighbors=5)
            for (x, y, z, w) in detection:
                cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)

            if detection != ():
                cv2.imwrite(detections_path + "/detection" + str(frame_count) + ".png", frame)
                frame_count += 1

            capture_to_write.write(frame)
            cv2.putText(frame, "PRESS SPACE TO STOP RECORDING", (10, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (20, 226, 20), 1)
            cv2.imshow("RECORDING", frame)
            cv2.namedWindow("RECORDING")

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    retry = False
                    cv2.destroyAllWindows()
                    video.release()
                    self.manager.transition = SlideTransition(direction="left")
                    self.manager.current = "ProcessedScreen"

    def _scan_and_save_emotion_result(self, capture_to_write, video, width, height):
        create_detections_directory_if_not_exists()
        retry = True
        frame_count = 0
        while retry:
            _, frame_raw = video.read()
            frame = cv2.resize(frame_raw, (int(width), int(height)))
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_classifier = cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_frontalface_default.xml")
            detection = face_classifier.detectMultiScale(frame, scaleFactor=model_scale_factor, minNeighbors=5)
            for (x, y, z, w) in detection:
                cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                gray_image_cropped = gray_image[y: y + w, x: x + z]
                resized_gray_image_cropped = cv2.resize(gray_image_cropped, (48, 48))
                gray_image_array = ima.img_to_array(resized_gray_image_cropped)
                gray_image_array_expanded = numpy.expand_dims(gray_image_array, axis=0)
                gray_image_array_expanded /= 255
                prediction = self.loaded_model.predict(gray_image_array_expanded)
                final_prediction = numpy.argmax(prediction[0])
                result = list_of_emotions[final_prediction]
                if z >= 210 and w >= 210:
                    cv2.putText(frame, result, (int(x + 5), int(y + 45)), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                (0, 0, 255), 2)
                else:
                    cv2.putText(frame, result, (int(x + 5), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (0, 0, 255), 1)

            if detection != ():
                cv2.imwrite(detections_path + "/detection" + str(frame_count) + ".png", frame)
                frame_count += 1

            capture_to_write.write(frame)
            cv2.putText(frame, "PRESS SPACE TO STOP RECORDING", (10, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (20, 226, 20), 1)
            cv2.imshow("RECORDING", frame)
            cv2.namedWindow("RECORDING")

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    retry = False
                    cv2.destroyAllWindows()
                    video.release()
                    self.manager.transition = SlideTransition(direction="left")
                    self.manager.current = "ProcessedScreen"

    def _scan_and_save_body_part_cropped_result(self, capture_to_write, video):
        retry = True
        frame_count = 0
        while retry:
            _, frame_raw = video.read()
            frame = cv2.resize(frame_raw, (520, 400))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            body_classifier = get_body_classifier()
            detection = body_classifier.detectMultiScale(gray_frame, scaleFactor=model_scale_factor, minNeighbors=5)

            create_detections_directory_if_not_exists()
            if detection == ():
                capture_to_write.write(frame)
                cv2.putText(frame, "PRESS SPACE ONCE OR TWICE TO STOP RECORDING", (10, 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (20, 226, 20), 1)
                cv2.putText(frame, "NO DETECTIONS", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (20, 226, 20), 1)
                cv2.imshow("RECORDING", frame)
                cv2.namedWindow("RECORDING")
            else:
                for (x, y, z, w) in detection:
                    cut = frame[y:y + w, x:x + z]
                    cut_image = cv2.resize(cut, (520, 400))
                    capture_to_write.write(cut_image)
                    cv2.imwrite(detections_path + "/detection" + str(frame_count) + ".png", cut_image)
                    cv2.putText(cut_image, "PRESS SPACE ONCE OR TWICE TO STOP RECORDING", (10, 12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (20, 226, 20), 1)
                    cv2.imshow("RECORDING", cut_image)
                    cv2.namedWindow("RECORDING")
                    frame_count += 1

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    retry = False
                    cv2.destroyAllWindows()
                    video.release()
                    self.manager.transition = SlideTransition(direction="left")
                    self.manager.current = "ProcessedScreen"

    def _scan_and_save_emotion_cropped_result(self, capture_to_write, video):
        create_detections_directory_if_not_exists()
        retry = True
        frame_count = 0
        while retry:
            _, frame_raw = video.read()
            frame = cv2.resize(frame_raw, (520, 400))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_classifier = cv2.CascadeClassifier(
                path_to_app_assets + "/haarcascade_frontalface_default.xml")
            detection = face_classifier.detectMultiScale(frame, scaleFactor=model_scale_factor, minNeighbors=5)

            if detection == ():
                capture_to_write.write(frame)
                cv2.putText(frame, "PRESS SPACE ONCE OR TWICE TO STOP RECORDING", (10, 12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (20, 226, 20), 1)
                cv2.putText(frame, "NO DETECTIONS", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (20, 226, 20), 1)
            else:
                for (x, y, z, w) in detection:
                    image_cropped = frame[y: y + w, x: x + z]
                    image_result = cv2.resize(image_cropped, (520, 400))
                    gray_frame_cropped = gray_frame[y: y + w, x: x + z]
                    resized_gray_frame_cropped = cv2.resize(gray_frame_cropped, (48, 48))
                    gray_frame_array = ima.img_to_array(resized_gray_frame_cropped)
                    gray_frame_array_expanded = numpy.expand_dims(gray_frame_array, axis=0)
                    gray_frame_array_expanded /= 255
                    prediction = self.loaded_model.predict(gray_frame_array_expanded)
                    final_prediction = numpy.argmax(prediction[0])
                    result = list_of_emotions[final_prediction]
                    cv2.putText(image_result, result, (10, 45),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.4,
                                (0, 0, 255), 2)
                    capture_to_write.write(image_result)
                    cv2.imwrite(detections_path + "/detection" + str(frame_count) + ".png", image_result)
                    cv2.putText(image_result, "PRESS SPACE ONCE OR TWICE TO STOP RECORDING", (10, 12),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (20, 226, 20), 1)
                    frame = image_result
                    frame_count += 1

            cv2.imshow("RECORDING", frame)
            cv2.namedWindow("RECORDING")

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    retry = False
                    cv2.destroyAllWindows()
                    video.release()
                    self.manager.transition = SlideTransition(direction="left")
                    self.manager.current = "ProcessedScreen"

    def _scan_and_show_emotion_result(self, video, width, height):
        retry = True
        while retry:
            _, frame_raw = video.read()
            frame = cv2.resize(frame_raw, (int(width), int(height)))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_classifier = cv2.CascadeClassifier(
                path_to_app_assets + "/haarcascade_frontalface_default.xml")
            detection = face_classifier.detectMultiScale(frame, scaleFactor=model_scale_factor, minNeighbors=5)

            if detection == ():
                cv2.putText(frame, "NO DETECTIONS", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (20, 226, 20), 1)
                for (x, y, z, w) in detection:
                    cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                    cv2.putText(frame, "PRESS SPACE ONCE/TWICE TO STOP SHOWING VIDEO", (10, 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (20, 226, 20), 1)
            else:
                cv2.putText(frame, "PRESS SPACE ONCE/TWICE TO STOP SHOWING VIDEO", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (20, 226, 20), 1)

                for (x, y, z, w) in detection:
                    cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                    gray_frame_cropped = gray_frame[y: y + w, x: x + z]
                    resized_gray_frame_cropped = cv2.resize(gray_frame_cropped, (48, 48))
                    gray_frame_array = ima.img_to_array(resized_gray_frame_cropped)
                    gray_frame_array_expanded = numpy.expand_dims(gray_frame_array, axis=0)
                    gray_frame_array_expanded /= 255
                    prediction = self.loaded_model.predict(gray_frame_array_expanded)
                    final_prediction = numpy.argmax(prediction[0])
                    result = list_of_emotions[final_prediction]
                    if z >= 210 and w >= 210:
                        cv2.putText(frame, result, (int(x + 5), int(y + 45)), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.4,
                                    (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, result, (int(x + 5), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35,
                                    (0, 0, 255), 1)

            cv2.imshow("PREVIEW", frame)
            cv2.namedWindow("PREVIEW")

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        retry = False
                        cv2.destroyAllWindows()

    def _scan_and_show_emotion_cropped_result(self, video):
        retry = True
        while retry:
            _, frame_raw = video.read()
            frame = cv2.resize(frame_raw, (520, 400))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_classifier = cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_frontalface_default.xml")
            detection = face_classifier.detectMultiScale(frame, scaleFactor=model_scale_factor, minNeighbors=5)

            if detection == ():
                cv2.putText(frame, "PRESS SPACE ONCE OR TWICE TO STOP SHOWING VIDEO", (10, 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (20, 226, 20), 1)
                cv2.putText(frame, "NO DETECTIONS", (140, 205),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (20, 226, 20), 1)
            else:
                for (x, y, z, w) in detection:
                    frame_cropped = frame[y: y + w, x: x + z]
                    frame_result = cv2.resize(frame_cropped, (520, 400))
                    gray_frame_cropped = gray_frame[y: y + w, x: x + z]
                    resized_gray_frame_cropped = cv2.resize(gray_frame_cropped, (48, 48))
                    gray_frame_array = ima.img_to_array(resized_gray_frame_cropped)
                    gray_frame_array_expanded = numpy.expand_dims(gray_frame_array, axis=0)
                    gray_frame_array_expanded /= 255
                    prediction = self.loaded_model.predict(gray_frame_array_expanded)
                    final_prediction = numpy.argmax(prediction[0])
                    result = list_of_emotions[final_prediction]
                    cv2.putText(frame_result, result, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.4,
                                (0, 0, 255), 2)
                    cv2.putText(frame_result, "PRESS SPACE TWICE TO CLOSE THE VIDEO", (10, 11),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.30,
                                (20, 226, 20), 1)
                    frame = frame_result

            cv2.imshow("PREVIEW", frame)
            cv2.namedWindow("PREVIEW")

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    retry = False
                    cv2.destroyAllWindows()
                    video.release()

    @staticmethod
    def _scan_and_show_body_part_result(video, width, height):
        retry = True
        while retry:
            _, frame_raw = video.read()
            frame = cv2.resize(frame_raw, (int(width), int(height)))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            body_classifier = get_body_classifier()
            detection = body_classifier.detectMultiScale(gray, scaleFactor=model_scale_factor, minNeighbors=5)
            if detection == ():
                cv2.putText(frame, "NO DETECTIONS", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (20, 226, 20), 1)
            else:
                for (x, y, z, w) in detection:
                    cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)

            cv2.putText(frame, "PRESS SPACE ONCE/TWICE TO STOP SHOWING VIDEO", (10, 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (20, 226, 20), 1)
            cv2.imshow("PREVIEW", frame)
            cv2.namedWindow("PREVIEW")

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        retry = False
                        cv2.destroyAllWindows()
                        video.release()

    @staticmethod
    def _scan_and_show_body_part_cropped_result(video):
        retry = True
        while retry:
            _, frame_raw = video.read()
            frame = cv2.resize(frame_raw, (520, 400))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            body_classifier = get_body_classifier()
            detection = body_classifier.detectMultiScale(gray_frame, scaleFactor=model_scale_factor, minNeighbors=5)

            if detection == ():
                cv2.putText(frame, "PRESS SPACE ONCE OR TWICE TO STOP SHOWING VIDEO", (10, 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (20, 226, 20), 1)
                cv2.putText(frame, "NO DETECTIONS", (140, 205),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (20, 226, 20), 1)
            else:
                for (x, y, z, w) in detection:
                    frame_cropped = frame[y:y + w, x:x + z]
                    resized_frame_cropped = cv2.resize(frame_cropped, (520, 400))
                    cv2.putText(resized_frame_cropped, "PRESS SPACE ONCE OR TWICE TO STOP SHOWING VIDEO", (10, 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (20, 226, 20), 1)
                    frame = resized_frame_cropped

            cv2.imshow("PREVIEW", frame)
            cv2.namedWindow("PREVIEW")

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    retry = False
                    cv2.destroyAllWindows()
                    video.release()


class ProcessedScreen(Screen):
    def __init__(self, **kwargs):
        super(ProcessedScreen, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        result_label = Label(
            text="$$$ YOUR PROCESSED FILE HAS BEEN SAVED IN THE DIRECTORY OF THE APP $$$",
            pos_hint={"x": 0, "y": 0.05}, color=(0.309, 0.933, 0.078, 4), font_size="12.5sp")
        acknowledge_button = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.30, "y": 0.32},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.4, 0.10))
        acknowledge_button.bind(on_press=self.switch_to_welcome_screen)
        logo_image = Image(source=path_to_app_assets + "/logo2.png", size_hint=(0.15, .3), pos_hint={"x": 0.423, "y": 0.65})

        layout.add_widget(result_label)
        layout.add_widget(acknowledge_button)
        layout.add_widget(logo_image)
        self.add_widget(layout)

    def switch_to_welcome_screen(self, *args):
        self.manager.transition = SlideTransition(direction="left")
        self.manager.current = "WelcomeScreen"


class NoDetectionsIdentifiedOnImageScreen(Screen):
    def __init__(self, **kwargs):
        super(NoDetectionsIdentifiedOnImageScreen, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        no_detections_label = Label(text="NO DETECTIONS IDENTIFIED",
                       pos_hint={"x": 0, "y": 0.09}, color=(0.309, 0.933, 0.078, 4), font_size="16sp")
        acknowledge_button = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.35, "y": 0.4},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.10))
        acknowledge_button.bind(on_press=self.switch_to_image_selection_screen)

        layout.add_widget(no_detections_label)
        layout.add_widget(acknowledge_button)
        self.add_widget(layout)

    def switch_to_image_selection_screen(self, *args):
        self.manager.transition = SlideTransition(direction="up")
        self.manager.current = "ChooseImageScreen"


class NoDetectionsIdentifiedOnImageCamScreen(Screen):
    def __init__(self, **kwargs):
        super(NoDetectionsIdentifiedOnImageCamScreen, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        no_detection_label = Label(text="NO DETECTIONS IDENTIFIED",
                       pos_hint={"x": 0, "y": 0.09}, color=(0.309, 0.933, 0.078, 4), font_size="16sp")
        acknowledge_button = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.35, "y": 0.4},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.3, 0.10))
        acknowledge_button.bind(on_press=self.switch_to_image_cam_selection_screen)

        layout.add_widget(no_detection_label)
        layout.add_widget(acknowledge_button)
        self.add_widget(layout)

    def switch_to_image_cam_selection_screen(self, *args):
        self.manager.transition = SlideTransition(direction="up")
        self.manager.current = "ChooseImageCamInputScreen"


class ErrorImageCamScreen(Screen):
    def __init__(self, **kwargs):
        super(ErrorImageCamScreen, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        error_label = Label(text="ERROR: CAMERA FAILED TO INITIALIZE",
                       pos_hint={"x": 0, "y": 0.09}, color=(0.309, 0.933, 0.078, 4), font_size="13sp")
        acknowledge_button = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.30, "y": 0.4},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.4, 0.10))
        acknowledge_button.bind(on_press=self.switch_to_image_cam_selection_screen)

        layout.add_widget(error_label)
        layout.add_widget(acknowledge_button)
        self.add_widget(layout)

    def switch_to_image_cam_selection_screen(self, *args):
        self.manager.transition = SlideTransition(direction="up")
        self.manager.current = "ChooseImageCamInputScreen"


class ErrorVideoCamScreen(Screen):
    def __init__(self, **kwargs):
        super(ErrorVideoCamScreen, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        error_label = Label(text="ERROR: EITHER NAME/TYPE OF YOUR FOLDER/FILE IS WRONG OR FOLDER/FILE DOESN'T EXIST",
                       pos_hint={"x": 0, "y": 0.09}, color=(0.309, 0.933, 0.078, 4), font_size="13sp")
        acknowledge_button = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.30, "y": 0.4},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.4, 0.10))
        acknowledge_button.bind(on_press=self.switch_to_video_cam_selection_screen)

        layout.add_widget(error_label)
        layout.add_widget(acknowledge_button)
        self.add_widget(layout)

    def switch_to_video_cam_selection_screen(self, *args):
        self.manager.transition = SlideTransition(direction="up")
        self.manager.current = "ChooseVideoCamInputScreen"


class ErrorImageSelectionScreen(Screen):
    def __init__(self, **kwargs):
        super(ErrorImageSelectionScreen, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        error_label = Label(text="ERROR: EITHER NAME/TYPE OF YOUR FOLDER/FILE IS WRONG OR FOLDER/FILE DOESN'T EXIST",
                       pos_hint={"x": 0, "y": 0.09}, color=(0.309, 0.933, 0.078, 4), font_size="13sp")
        acknowledge_button = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.30, "y": 0.4},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.4, 0.10))
        acknowledge_button.bind(on_press=self.switch_to_image_selection_screen)

        layout.add_widget(error_label)
        layout.add_widget(acknowledge_button)
        self.add_widget(layout)

    def switch_to_image_selection_screen(self, *args):
        self.manager.transition = SlideTransition(direction="up")
        self.manager.current = "ChooseImageScreen"


class ErrorVideoSelectionScreen(Screen):
    def __init__(self, **kwargs):
        super(ErrorVideoSelectionScreen, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))

        error_label = Label(text="ERROR: EITHER NAME/TYPE OF YOUR FOLDER/FILE IS WRONG OR FOLDER/FILE DOESN'T EXIST",
                       pos_hint={"x": 0, "y": 0.09}, color=(0.309, 0.933, 0.078, 4), font_size="13sp")
        acknowledge_button = Button(text="OK", background_color=(0.309, 0.933, 0.078, 4), pos_hint={"x": 0.30, "y": 0.4},
                      color=(0.141, 0.054, 0.078, 4), size_hint=(0.4, 0.10))
        acknowledge_button.bind(on_press=self.switch_to_video_selection_screen)

        layout.add_widget(error_label)
        layout.add_widget(acknowledge_button)
        self.add_widget(layout)

    def switch_to_video_selection_screen(self, *args):
        self.manager.transition = SlideTransition(direction="up")
        self.manager.current = "ChooseVideoScreen"


class LoadingAndSavingScannedVideoScreen(Screen):
    def __init__(self, **kwargs):
        super(LoadingAndSavingScannedVideoScreen, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))
        loading_image = AsyncImage(source=path_to_app_assets + '/loading.gif',
                          size_hint=(0.15, 1 / 4),
                          keep_ratio=False,
                          allow_stretch=True,
                          pos_hint={'x': 0.435, 'y': 0.4})
        processing_label = Label(text="PROCESSING", pos_hint={"x": 0.005, "y": 0.20}, color=(0.309, 0.933, 0.078, 4))
        processing_note_label = Label(text="30 seconds of video get processed approximately for 1 minute 15 seconds",
                       pos_hint={"x": 0.005, "y": -0.3}, color=(0.309, 0.933, 0.078, 4))
        layout.add_widget(loading_image)
        layout.add_widget(processing_label)
        layout.add_widget(processing_note_label)
        self.add_widget(layout)
        self.loaded_model = ""

    def on_enter(self, *args):
        try:
            self.loaded_model = model_from_json(open(path_to_app_assets + "/neuralnet.json", "r").read())
            self.loaded_model.load_weights(path_to_app_assets + "/weights.h5")
            audio = AudioFileClip(selected_file_path)
            audio.write_audiofile(path_to_app + "/audio.mp3")
            video = cv2.VideoCapture(selected_file_path)
            height_value = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width_value = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            width = str(width_value)
            height = str(height_value)
            width = float(width)
            height = float(height)
            capture_to_write = cv2.VideoWriter(path_to_app + "/ProcessedVideo.avi", cv2.VideoWriter_fourcc(*"XVID"), 30,
                                      (int(width), int(height)))
            try:
                if area_for_scan != "emotion":
                    self._scan_and_write_video_for_body_parts(capture_to_write, video, height, width)
                self._scan_and_write_video_for_emotion(capture_to_write, video, height, width)
            except Exception as e:
                # at the end of the video exception will be thrown so it is only a warning
                print("Warning: " + str(e))
                video.release()
                self._try_combining_audio_and_video()
                self.manager.transition = SlideTransition(direction="left")
                self.manager.current = "ProcessedScreen"
        except Exception as e:
            print("Error: " + str(e))
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "ErrorVideoSelectionScreen"

    def _scan_and_write_video_for_emotion(self, capture_to_write, video, height, width):
        create_detections_directory_if_not_exists()
        frame_count = 0
        while True:
            _, frame_raw = video.read()
            frame = cv2.resize(frame_raw, (int(width), int(height)))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_recognition_classifier = cv2.CascadeClassifier(path_to_app_assets + "/haarcascade_frontalface_default.xml")
            detection = face_recognition_classifier.detectMultiScale(frame, scaleFactor=model_scale_factor, minNeighbors=5)
            for (x, y, z, w) in detection:
                cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)
                cut_gray = gray[y: y + w, x: x + z]
                resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                gray_image_array = ima.img_to_array(resized_cut_gray)
                gray_image_array_expanded = numpy.expand_dims(gray_image_array, axis=0)
                gray_image_array_expanded /= 255
                prediction = self.loaded_model.predict(gray_image_array_expanded)
                final_prediction = numpy.argmax(prediction[0])
                result = list_of_emotions[final_prediction]
                if z >= 210 and w >= 210:
                    cv2.putText(frame, result, (int(x + 5), int(y + 45)), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                                (0, 0, 255), 2)
                    continue
                cv2.putText(frame, result, (int(x + 5), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (0, 0, 255), 1)
            capture_to_write.write(frame)

            if detection != ():
                cv2.imwrite(detections_path + "/detection" + str(frame_count) + ".png", frame)
                frame_count += 1

    @staticmethod
    def _scan_and_write_video_for_body_parts(capture_to_write, video, height, width):
        create_detections_directory_if_not_exists()
        frame_count = 0
        while True:
            _, frame_raw = video.read()
            frame = cv2.resize(frame_raw, (int(width), int(height)))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            body_classifier = get_body_classifier()
            detection = body_classifier.detectMultiScale(gray_frame, scaleFactor=model_scale_factor, minNeighbors=5)
            for (x, y, z, w) in detection:
                cv2.rectangle(frame, (x, y), (x + z, y + w), (20, 226, 20), thickness=7)

            if detection != ():
                cv2.imwrite(detections_path + "/detection" + str(frame_count) + ".png", frame)
                frame_count += 1

            capture_to_write.write(frame)

    # combination of video and audio might not work if installed ffmpeg version is invalid
    @staticmethod
    def _try_combining_audio_and_video():
        try:
            input_audio = ffmpeg.input(path_to_app + "/audio.mp3")
            input_video = ffmpeg.input(path_to_app + "/ProcessedVideo.avi")
            (
                ffmpeg
                    .concat(input_video, input_audio, v=1, a=1)
                    .output(path_to_app + "/ProcessedVideoWithAudio.avi")
                    .global_args('-loglevel', 'quiet')
                    .run(capture_stdout=True, overwrite_output=True)
            )
        except Exception as ffmpeg_e:
            print("FFMPEG Error: " + str(ffmpeg_e))


class LoadingAndSavingScannedVideoCroppedScreen(Screen):
    def __init__(self, **kwargs):
        super(LoadingAndSavingScannedVideoCroppedScreen, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))
        loading_image = AsyncImage(source=path_to_app_assets + "/loading.gif",
                          size_hint=(0.15, 1 / 4),
                          keep_ratio=False,
                          allow_stretch=True,
                          pos_hint={"x": 0.435, "y": 0.4})
        processing_label = Label(text="PROCESSING", pos_hint={"x": 0.005, "y": 0.20}, color=(0.309, 0.933, 0.078, 4))
        processing_note_label = Label(text="30 seconds of video get processed approximately for 1 minute 15 seconds",
                       pos_hint={"x": 0.005, "y": -0.3}, color=(0.309, 0.933, 0.078, 4))
        layout.add_widget(loading_image)
        layout.add_widget(processing_label)
        layout.add_widget(processing_note_label)
        self.add_widget(layout)
        self.loaded_model = ""

    def on_enter(self, *args):
        try:
            self.loaded_model = model_from_json(open(path_to_app_assets + "/neuralnet.json", "r").read())
            self.loaded_model.load_weights(path_to_app_assets + "/weights.h5")
            selected_file_extension = str(selected_file_path[-3:])
            if selected_file_extension.lower() not in invalid_file_extensions:
                video = cv2.VideoCapture(selected_file_path)

                # check if selected video can be resized
                _, frame_raw_check = video.read()
                cv2.resize(frame_raw_check, (520, 400))
                video.release()

                video = cv2.VideoCapture(selected_file_path)
                capture_to_write = cv2.VideoWriter(path_to_app + "/ProcessedVideo.avi", cv2.VideoWriter_fourcc(*"XVID"), 30,
                                          (520, 400))
                try:
                    if area_for_scan != "emotion":
                        self._scan_and_write_result_for_body_parts(capture_to_write, video)
                    self._scan_and_write_result_for_emotion(capture_to_write, video)
                except Exception as e:
                    # at the end of the video exception will be thrown so it is only a warning
                    print("Warning: " + str(e))
                    video.release()
                    self.manager.transition = SlideTransition(direction="left")
                    self.manager.current = "ProcessedScreen"
                    return
            else:
                print("Error: selected file's extension " + selected_file_extension + " is not valid")
                self.manager.transition = SlideTransition(direction="down")
                self.manager.current = "ErrorVideoSelectionScreen"
        except Exception as e:
            print("Error: " + str(e))
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "ErrorVideoSelectionScreen"

    def _scan_and_write_result_for_emotion(self, capture_to_write, video):
        create_detections_directory_if_not_exists()
        frame_count = 0
        while True:
            _, frame_raw = video.read()
            frame = cv2.resize(frame_raw, (520, 400))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_classifier = cv2.CascadeClassifier(
                path_to_app_assets + "/haarcascade_frontalface_default.xml")
            detection = face_classifier.detectMultiScale(gray_frame, scaleFactor=model_scale_factor, minNeighbors=5)

            if detection == ():
                capture_to_write.write(frame)
                continue

            for (x, y, z, w) in detection:
                image_cropped = frame[y: y + w, x: x + z]
                image_final = cv2.resize(image_cropped, (520, 400))
                gray_image_cropped = gray_frame[y: y + w, x: x + z]
                resized_gray_image_cropped = cv2.resize(gray_image_cropped, (48, 48))
                gray_image_array = ima.img_to_array(resized_gray_image_cropped)
                gray_image_array_expanded = numpy.expand_dims(gray_image_array, axis=0)
                gray_image_array_expanded /= 255
                prediction = self.loaded_model.predict(gray_image_array_expanded)
                final_prediction = numpy.argmax(prediction[0])
                result = list_of_emotions[final_prediction]
                cv2.putText(image_final, result, (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.4,
                            (0, 0, 255), 2)
                capture_to_write.write(image_final)
                cv2.imwrite(detections_path + "/detection" + str(frame_count) + ".png", frame)
                frame_count += 1

    @staticmethod
    def _scan_and_write_result_for_body_parts(capture_to_write, video):
        frame_count = 0
        while True:
            _, frame_raw = video.read()
            frame = cv2.resize(frame_raw, (520, 400))
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            body_classifier = get_body_classifier()
            detection = body_classifier.detectMultiScale(gray_frame, scaleFactor=model_scale_factor, minNeighbors=5)

            create_detections_directory_if_not_exists()
            for (x, y, z, w) in detection:
                final_frame = frame[y:y + w, x:x + z]
                final_cut_frame = cv2.resize(final_frame, (520, 400))
                capture_to_write.write(final_cut_frame)
                cv2.imwrite(detections_path + "/detection" + str(frame_count) + ".png", final_cut_frame)
                frame_count += 1


class LoadingAndShowingResultVideoCroppedScreen(Screen):
    def __init__(self, **kwargs):
        super(LoadingAndShowingResultVideoCroppedScreen, self).__init__(**kwargs)
        layout = FloatLayout(size=(350, 600))
        loading_image = AsyncImage(source=path_to_app_assets + "/loading.gif",
                          size_hint=(0.15, 1 / 4),
                          keep_ratio=False,
                          allow_stretch=True,
                          pos_hint={"x": 0.435, "y": 0.4})
        processing_label = Label(text="PROCESSING", pos_hint={"x": 0.005, "y": 0.20}, color=(0.309, 0.933, 0.078, 4))
        processing_note_label = Label(text="30 seconds of video get processed approximately for 1 minute 15 seconds",
                       pos_hint={"x": 0.005, "y": -0.3}, color=(0.309, 0.933, 0.078, 4))
        layout.add_widget(loading_image)
        layout.add_widget(processing_label)
        layout.add_widget(processing_note_label)
        self.add_widget(layout)
        self.show_final_result = True
        self.loaded_model = ""

    def on_enter(self, *args):
        try:
            self.loaded_model = model_from_json(open(path_to_app_assets + "/neuralnet.json", "r").read())
            self.loaded_model.load_weights(path_to_app_assets + "/weights.h5")
            selected_file_extension = str(selected_file_path[-3:])
            if selected_file_extension.lower() not in invalid_file_extensions:
                video = cv2.VideoCapture(selected_file_path)

                # check if selected video can be resized
                _, frame_raw_check = video.read()
                cv2.resize(frame_raw_check, (520, 400))
                video.release()

                video = cv2.VideoCapture(selected_file_path)
                capture = cv2.VideoWriter(path_to_app_assets + "/detection.avi", cv2.VideoWriter_fourcc(*"XVID"), 30,
                                          (520, 400))
                try:
                    if area_for_scan != "emotion":
                        self._scan_and_write_for_body_parts(capture, video)
                    self._scan_and_write_for_emotion(capture, video)
                except Exception as e:
                    # at the end of the video exception will be thrown so it is only a warning
                    print("Warning: " + str(e))
                    video.release()
                    self.manager.transition = SlideTransition(direction="left")
                    self.manager.current = "ChooseVideoScreen"
                    return
            else:
                print("Error: selected file's extension " + selected_file_extension + " is not valid")
                self.show_final_result = False
                self.manager.transition = SlideTransition(direction="down")
                self.manager.current = "ErrorVideoSelectionScreen"
        except Exception as e:
            print("Error: " + str(e))
            self.show_final_result = False
            self.manager.transition = SlideTransition(direction="down")
            self.manager.current = "ErrorVideoSelectionScreen"

    def on_leave(self, *args):
        if self.show_final_result:
            detected = ""
            try:
                retry = True
                detected = cv2.VideoCapture(path_to_app_assets + "/detection.avi")
                while retry:
                    _, frame = detected.read()
                    cv2.imshow("PREVIEW", frame)
                    cv2.namedWindow("PREVIEW")
                    time.sleep(0.1)
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                            retry = False
                            cv2.destroyAllWindows()
                return
            except Exception as e:
                print("Error: " + str(e))
                detected.release()
                cv2.destroyAllWindows()
                return
        print("Final result not shown due to error")

    def _scan_and_write_for_emotion(self, capture_to_write, video):
        while True:
            _, frame_raw = video.read()
            frame = cv2.resize(frame_raw, (520, 400))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_classifier = cv2.CascadeClassifier(
                path_to_app_assets + "/haarcascade_frontalface_default.xml")
            detection = face_classifier.detectMultiScale(frame, scaleFactor=model_scale_factor, minNeighbors=5)
            for (x, y, z, w) in detection:
                image_cropped = frame[y: y + w, x: x + z]
                image_result = cv2.resize(image_cropped, (520, 400))
                cut_gray = gray[y: y + w, x: x + z]
                resized_cut_gray = cv2.resize(cut_gray, (48, 48))
                image_gray_array = ima.img_to_array(resized_cut_gray)
                image_gray_array_expended = numpy.expand_dims(image_gray_array, axis=0)
                image_gray_array_expended /= 255
                prediction = self.loaded_model.predict(image_gray_array_expended)
                final_prediction = numpy.argmax(prediction[0])
                result = list_of_emotions[final_prediction]
                cv2.putText(image_result, result, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.4,
                            (0, 0, 255), 2)
                cv2.putText(image_result, "PRESS SPACE TWICE TO CLOSE THE VIDEO", (10, 11),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.30,
                            (20, 226, 20), 1)
                capture_to_write.write(image_result)

    @staticmethod
    def _scan_and_write_for_body_parts(capture_to_write, video):
        while True:
            _, frame_raw = video.read()
            frame = cv2.resize(frame_raw, (520, 400))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            body_classifier = get_body_classifier()
            detection = body_classifier.detectMultiScale(gray, scaleFactor=model_scale_factor, minNeighbors=5)
            for (x, y, z, w) in detection:
                final_frame = frame[y:y + w, x:x + z]
                final_cut_frame = cv2.resize(final_frame, (520, 400))
                cv2.putText(final_cut_frame, "PRESS SPACE ONCE OR TWICE TO CLOSE THE VIDEO", (10, 11),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.30,
                            (20, 226, 20), 1)
                capture_to_write.write(final_cut_frame)


def set_file_paths():
    global path_to_app
    global path_to_app_assets
    global detections_path
    if getattr(sys, 'frozen', False):
        path_to_app = str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))))
        path_to_app_assets = str(os.path.dirname(sys.executable)) + "/assets"
    elif __file__:
        path_to_app = str(os.path.dirname(__file__))
        path_to_app_assets = str(os.path.dirname(__file__)) + "/assets"
    detections_path = path_to_app + "/detections"


def set_config():
    pygame.init()

    Config.set("kivy", "window_icon", path_to_app_assets + "/logo.png")
    Config.set('graphics', 'resizable', False)
    Config.set("graphics", "width", "850")
    Config.set("graphics", "height", "630")
    Config.write()


def init_screen_manager():
    manager = ScreenManager()
    manager.add_widget(WelcomeScreen(name="WelcomeScreen"))
    manager.add_widget(ChooseInputScreen(name="ChooseInputScreen"))
    manager.add_widget(ChooseImageScreen(name="ChooseImageScreen"))
    manager.add_widget(ChooseImageCamInputScreen(name="ChooseImageCamInputScreen"))
    manager.add_widget(ChooseVideoScreen(name="ChooseVideoScreen"))
    manager.add_widget(ChooseVideoCamInputScreen(name="ChooseVideoCamInputScreen"))
    manager.add_widget(NoDetectionsIdentifiedOnImageScreen(name="NoDetectionsIdentifiedOnImageScreen"))
    manager.add_widget(NoDetectionsIdentifiedOnImageCamScreen(name="NoDetectionsIdentifiedOnImageCamScreen"))
    manager.add_widget(LoadingAndSavingScannedVideoScreen(name="LoadingAndSavingScannedVideoScreen"))
    manager.add_widget(LoadingAndSavingScannedVideoCroppedScreen(name="LoadingAndSavingScannedVideoCroppedScreen"))
    manager.add_widget(LoadingAndShowingResultVideoCroppedScreen(name="LoadingAndShowingResultVideoCroppedScreen"))
    manager.add_widget(ProcessedScreen(name="ProcessedScreen"))
    manager.add_widget(ErrorImageCamScreen(name="ErrorImageCamScreen"))
    manager.add_widget(ErrorVideoSelectionScreen(name="ErrorVideoSelectionScreen"))
    manager.add_widget(ErrorVideoCamScreen(name="ErrorVideoCamScreen"))
    manager.add_widget(ErrorImageSelectionScreen(name="ErrorImageSelectionScreen"))
    return manager


class SquareFace(App):
    def __init__(self):
        super().__init__()
        set_file_paths()
        set_config()

    def build(self):
        return init_screen_manager()


if __name__ == "__main__":
    SquareFace().run()
