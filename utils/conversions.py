def convert_pixels_to_meters(pixel_distance,reference_distance_meters,reference_distance_pixels):
    return (pixel_distance * reference_distance_meters)/ reference_distance_pixels

def convert_meters_to_pixels(meters, reference_distance_in_meters,reference_distance_in_pixels):
    return (meters*reference_distance_in_pixels) / reference_distance_in_meters