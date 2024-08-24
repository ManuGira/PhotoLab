from exif import Image as ExifImage
from PIL import Image as PilImage
from PIL.ExifTags import TAGS


def main():
    # filename = "images/vermeer_758x640.jpg"
    filename = "images/etiquette adamiel.jpg"

    # Create ExifImage instance
    with open(filename, 'rb') as img_file:
        eimg = ExifImage(img_file)

    if not eimg.has_exif:
        print("Exif not found. Image must be jpg")
    else:
        print("Exif data listed by exif package")
        for tag in eimg.list_all():
            print(f"\t{tag}: {eimg.get(tag)}")
    print()

    # Create PILImage instance
    pimage = PilImage.open(filename)

    print("image info read by PIL:")
    print(pimage.info)
    print()

    # extract EXIF data with PIL
    exifdata = pimage.getexif()

    # iterating over all EXIF data fields
    print("Exifs data listed by PIL package")
    for tag_id in exifdata:
        # get the tag name, instead of human unreadable tag id
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        # decode bytes
        if isinstance(data, bytes):
            data = data.decode()
        print(f"\t{tag_id} ({tag}): {data}")

    # print()
    # print("save image to 300dpi")
    # pimage.save("images/etiquette adamiel 300dpi.jpg", dpi=(300, 300))

    # print("Set dpi to 300...")
    # pimage.info['dpi'] = (300, 300)
    # pimage.info['jfif_density'] = (300, 300)
    #
    #

if __name__ == '__main__':
    main()

