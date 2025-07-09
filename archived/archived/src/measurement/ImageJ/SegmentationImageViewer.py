



class SegmentationImageViewer(AbstractImageViewer):
    def _load_labels(self, image_name):
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                lines = file.readlines()
                labels = []
                for line in lines:
                    parts = line.split()
                    keypoints = [(float(parts[i]), float(parts[i + 1])) for i in range(5, len(parts), 2)]
                    labels.append(keypoints)
                return labels
        return []

    def _draw_annotations(self, image):
        labels = self._load_labels(self.current_image_name)
        for points in labels:
            self._draw_single_annotation(image, points)
        self._draw_imagej_results(image)

    def _draw_single_annotation(self, image, points):
        if len(points) > 2:
            pixel_points = self._convert_to_pixels(points, image.shape[1], image.shape[0])
            cv2.polylines(image, [np.array(pixel_points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    def _convert_to_pixels(self, normalized_points, image_width, image_height):
        return [(int(x * image_width), int(y * image_height)) for x, y in normalized_points]

    def _draw_imagej_results(self, image):
        imagej_result = self.imagej_results[self.imagej_results['Label'].str.contains(self.current_image_name)].iloc[0]
        bx, by = int(imagej_result['BX']), int(imagej_result['BY'])
        width, height = int(imagej_result['Width']), int(imagej_result['Height'])
        length = imagej_result['Length']
        diagonal_start = (bx, by)
        diagonal_end = (bx + width, by + height)
        cv2.line(image, diagonal_start, diagonal_end, (255, 0, 0), 2)
        cv2.putText(image, f'Length: {length:.2f}', (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
