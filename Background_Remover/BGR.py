from rembg import remove
from PIL import Image
import io

def remove_background(input_image_path, output_image_path):
    # 이미지를 읽어서 배경을 제거
    with open(input_image_path, 'rb') as input_file:
        input_data = input_file.read()
        output_data = remove(input_data)
    
    # 배경이 제거된 이미지를 저장
    with open(output_image_path, 'wb') as output_file:
        output_file.write(output_data)

# 사용 예시
input_image = 'Background_Remover\Test_1.png'  # 입력 이미지 경로
output_image = 'Test_1_BGR.png'  # 출력 이미지 경로 (PNG 형식이 투명 배경을 지원함)

remove_background(input_image, output_image)
