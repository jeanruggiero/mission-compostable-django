from skimage import io, filters, color
import numpy as np
from skimage.color import rgb2gray
from collections import deque


def find_lightest_area_value(filename, area_size):
    image = io.imread(filename)
    grayscale = rgb2gray(image)
    threshold = 0.01
    pixel_count = 0
    visited = np.zeros(grayscale.shape, dtype=np.int8)

    max_value = 0
    for y, y_val in enumerate(grayscale):
        for x, x_val in enumerate(y_val):
            if x_val > max_value:
                max_value = x_val

    while pixel_count < area_size:
        filtered = np.zeros(grayscale.shape, dtype=np.int8)
        count = 0
        for y, y_val in enumerate(grayscale):
            for x, x_val in enumerate(y_val):
                if x_val > max_value - threshold:
                    count += 1
                    filtered[y, x] = 1

        for y, y_val in enumerate(filtered):
            for x, x_val in enumerate(y_val):
                if x_val == 1:
                    start = (y, x)
                    visited = np.zeros(grayscale.shape, dtype=np.int8)
                    search_queue = deque()
                    search_queue.append(start)
                    visited[start] = 1
                    pixel_count = 1

                    while search_queue:
                        popped = search_queue.popleft()
                        up = (popped[0] + 1, popped[1])
                        down = (popped[0] - 1, popped[1])
                        left = (popped[0], popped[1] - 1)
                        right = (popped[0], popped[1] + 1)
                        for neighbor in [up, down, left, right]:
                            if filtered[neighbor] == 1 and visited[neighbor] == 0:
                                pixel_count += 1
                                visited[neighbor] = 1
                                search_queue.append(neighbor)
                                filtered[neighbor] = 0
                    if pixel_count >= area_size:
                        break
            if pixel_count >= area_size:
                break
        threshold += 0.001

    value_sum = 0
    value_count = 0
    for y, y_val in enumerate(grayscale):
        for x, x_val in enumerate(y_val):
            if visited[y][x] == 1:
                value_count += 1
                value_sum += grayscale[y][x]
    return value_sum / value_count


def find_mean_brightness(filename):
    image = io.imread(filename)
    blur = color.rgb2gray(image)
    blur = filters.gaussian(blur, sigma=np.std(blur), multichannel=False)
    t = filters.threshold_otsu(blur)
    mask = blur > t
    sel = np.zeros_like(image)
    sel[mask] = image[mask]
    grayscale = rgb2gray(sel)
    rows,cols = grayscale.shape[0:2]

    listy = []
    for x in range (1,rows):
        for y in range (1,cols):
            pixel = grayscale[x,y]
            if pixel > 0.0:
                listy.append(pixel)

    mean = np.mean(listy)
    return mean, np.max(listy)


def process_image(filename):
    max_area = find_lightest_area_value(filename, 10)
    mean, max_brightness = find_mean_brightness(filename)
    print((max_area - mean) / max_brightness)
    return (max_area - mean) / max_brightness


def is_compostable(filename):
    s_factor = process_image(filename)
    return True if s_factor < 0.2 else False
