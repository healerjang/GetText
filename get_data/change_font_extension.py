from fontTools.ttLib import TTFont
import os

# html에서만 사용되는 woff 형태의 폰트 확장자나 woff2 확장자를 ttf 확장자로 변환하는 코드

def change_font_extension(save_path):
    directory, filename = os.path.split(save_path)
    name, _ = os.path.splitext(filename)
    try:
        font = TTFont(save_path)
        font.flavor = None
        new_path = os.path.join(directory, f"{name}.ttf")
        font.save(new_path)
        print(f"변환 성공: {new_path}")
        os.remove(save_path)
    except Exception as e:
        print(f"변환 실패: {save_path} - {e}")

def main():
    fonts_path = "C:/SpliceImageTextData/fonts"
    for font_path in os.listdir(fonts_path):
        _, extension = os.path.splitext(font_path)
        if extension.lower() != ".ttf":
            change_font_extension(os.path.join(fonts_path, font_path))

if __name__ == "__main__":
    main()
