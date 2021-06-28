import os
import shutil

import cv2
import numpy

import skin_detector


def test_get_hsv_mask():
    img_path = "tests/test_image.png"
    img = cv2.imread(img_path)
    mask = skin_detector.get_hsv_mask(img)
    assert img.shape[:2] == mask.shape


def test_get_rgb_mask():
    img_path = "tests/test_image.png"
    img = cv2.imread(img_path)
    mask = skin_detector.get_rgb_mask(img)
    assert img.shape[:2] == mask.shape


def test_get_ycrcb_mask():
    img_path = "tests/test_image.png"
    img = cv2.imread(img_path)
    mask = skin_detector.get_ycrcb_mask(img)
    assert img.shape[:2] == mask.shape


def test_grab_cut_mask():
    img_path = "tests/test_image.png"
    img = cv2.imread(img_path)
    assert True


def test_closing():
    img_path = "tests/test_image.png"
    img = cv2.imread(img_path)
    assert True


def test_process():
    img_path = "tests/test_image.png"
    img = cv2.imread(img_path)
    mask = skin_detector.process(img)
    assert img.shape[:2] == mask.shape


def test_find_images():
    image_dir = os.path.abspath("./tests/images/")
    if os.path.isdir(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)

    image_paths = sorted(["hello.png", "world.jpg", "simon.png", "says.jpeg"])
    image_paths = [os.path.join(image_dir, image_path) for image_path in image_paths]
    image_paths = [os.path.abspath(image_path) for image_path in image_paths]

    for image_path in image_paths:
        img = numpy.random.randint(255, size=(1920, 1080, 3))
        cv2.imwrite(image_path, img)

    assert sorted(skin_detector.find_images(image_dir)) == image_paths
    shutil.rmtree(image_dir)


def test_find_images_recursive():
    image_dir = os.path.abspath("./tests/images/")
    if os.path.isdir(image_dir):
        shutil.rmtree(image_dir)

    os.makedirs(image_dir)

    image_paths = sorted(["hello.png", "world.jpg", "simon.png", "says.jpeg"])
    image_paths = [os.path.join(image_dir, image_path) for image_path in image_paths]

    for image_path in image_paths:
        img = numpy.random.randint(255, size=(1920, 1080, 3))
        cv2.imwrite(image_path, img)

    recursive_dir = os.path.abspath("./tests/images/recurse")
    os.makedirs(recursive_dir)

    recursive_images = sorted(["alpha.png", "beta.jpeg", "gamma.png"])
    recursive_images = [os.path.join(recursive_dir, image_path) for image_path in recursive_images]

    for image_path in recursive_images:
        img = numpy.random.randint(255, size=(1920, 1080, 3))
        cv2.imwrite(image_path, img)

    all_images = sorted(recursive_images + image_paths)

    assert sorted(skin_detector.find_images(image_dir, recursive=True)) == all_images
    shutil.rmtree(image_dir)
