import math


class Object_Tracker:
    """
    Class for tracking objects based on their bounding boxes.

    Attributes:
        center_points (dict): Dictionary containing object IDs as keys and their corresponding center points as values.
        id_count (int): Counter for assigning unique IDs to objects.
    """

    def __init__(self):
        """
        Initialize Object_Tracker with an empty dictionary for center points and ID count set to 0.
        """
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        """
        Update the object tracker based on the provided bounding box coordinates.

        Args:
            objects_rect (list): List of bounding box coordinates in the format [x, y, w, h].

        Returns:
            objects_bbs_ids (list): List of bounding box coordinates with assigned object IDs.
        """
        objects_bbs_ids = []

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            object_detected = False
            for obj_id, pt in self.center_points.items():
                # Calculate distance between the current center point and previously tracked center points
                dist = math.hypot(cx - pt[0], cy - pt[1])

                # If the distance is within a threshold, update the center point and assign the object ID
                if dist < 35:
                    self.center_points[obj_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, obj_id])
                    object_detected = True
                    break

            # If no existing object matches, assign a new object ID
            if not object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Update the center_points dictionary with the latest positions of tracked objects
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids
