!pip install ultralytics==8.0.196 roboflow


from roboflow import Roboflow
rf = Roboflow(api_key="Us3HvvWPK0OOkKDkROrX")
project = rf.workspace("djchoe").project("dj_choe")
version = project.version(3)
dataset = version.download("yolov8")
