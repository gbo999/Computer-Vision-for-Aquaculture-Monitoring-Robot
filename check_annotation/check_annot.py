import cv2

# Load your image
image_path = 'C:/Users/gbo10/Videos/research/counting_research_algorithms/check_annotation/GX010065_MP4-51-jpg_gamma_jpg.rf.cc7a25ee61769b282b89dd03802c032a.jpg'
image = cv2.imread(image_path)

# Your label data as a list of strings (each string is a line from your label file)
label_data = [
    "0 0.43125 0.334375 0.184375 0.29609375 0.39818616340361446 0.33247192101740297 2 0.42326925828313255 0.2834431057563588 2 0.39316955948795185 0.2314429049531459 2 0.4140721385542168 0.18687128514056223 2 0.4550411897590362 0.3191004016064257 2 0.4667466490963855 0.20469989959839358 2",
    "0 0.95390625 0.7703125 0.09140625 0.3546875 0.9583751694277108 0.7559022088353414 2 0.9642278614457831 0.7009305555555556 2 0.9266032379518071 0.6207016064257028 2 0.9123895143072287 0.6712161311914324 2 0.9801137612951807 0.6548731927710844 2 0.9533585090361445 0.6266445113788487 2",
    "0 0.35859375 0.615625 0.2046875 0.3609375 0.3179202936746988 0.5285869477911647 1 0.3354784450301205 0.5731585676037483 2 0.3831363516566265 0.5895015060240963 2 0.39484181099397586 0.7187592034805891 2 0.2886567018072289 0.6147587684069612 1 0.3020343373493976 0.7039020080321284 1"
]

# Image dimensions
height, width, _ = image.shape

for data in label_data:
    parts = data.split()
    # Extract bounding box coordinates
    x_center, y_center, bbox_width, bbox_height = [float(val) for val in parts[1:5]]
    x_center, y_center, bbox_width, bbox_height = int(x_center * width), int(y_center * height), int(bbox_width * width), int(bbox_height * height)
    top_left = (x_center - bbox_width // 2, y_center - bbox_height // 2)
    bottom_right = (x_center + bbox_width // 2, y_center + bbox_height // 2)

    # Draw bounding box
    cv2.rectangle(image, top_left, bottom_right, (255,0,0), 2)

    # Draw keypoints
    num_keypoints = (len(parts) - 5) // 3
    for i in range(num_keypoints):
        px, py = [float(val) for val in parts[5 + i*3: 7 + i*3]]  # Extract x, y coordinates
        px, py = int(px * width), int(py * height)  # Scale coordinates
        visibility = int(parts[7 + i*3])  # Extract visibility
        if visibility == 2:  # If the keypoint is visible
            cv2.circle(image, (px, py), 5, (0, 255, 0), -1)  # Draw the keypoint
            cv2.putText(image, str(i), (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label the keypoint with its index
# Display the result
            
            
cv2.imshow('Annotated Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
