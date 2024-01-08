from PIL import Image
import face_recognition

image = face_recognition.load_image_file("img.jpg")

face_locations = face_recognition.face_locations(image)

print("I found {} Face(s) in the photograph:".format(len(face_locations)))

for face_location in face_locations:
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {} Bottom: {} Left: {} Right: {}".format(top,bottom,left,right))

    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()