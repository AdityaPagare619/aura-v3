# Gallery App

This app provides access to the device's photo gallery and image files.

## Usage Examples

- "Find my grandma's photo"
- "Show me photos from last month"
- "List all images in the camera folder"
- "Find photos with beach"

## Capabilities

This app can:
- List image files from any directory
- Search files by name pattern
- Read image metadata (EXIF)
- Provide file paths for further processing

## When NOT to Use

- When you need to actually ANALYZE image content (use vision model)
- When image is in cloud but not downloaded locally
- When dealing with encrypted galleries

## Fallback Strategies

If gallery access fails:
1. Check WhatsApp/Media folder patterns
2. Check Telegram download folder
3. Check Downloads folder
4. Query chat apps for shared images
