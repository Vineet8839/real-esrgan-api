
# Real-ESRGAN Super-Resolution API

This API uses the Real-ESRGAN model to enhance images with super-resolution.

## Usage

POST `/enhance` with an image file to get an enhanced version.

## Example Request

```bash
curl -X POST "http://your-api-url/enhance" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
