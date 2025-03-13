from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import utils

class VisionLanguage:
    default_prompt = "what is this"
    def __init__(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(utils.get_vision_language_path())
        self.processor = AutoProcessor.from_pretrained(utils.get_vision_language_path())

    def image_request(self, prompt=default_prompt, image_path=None, url=None):
        content = []
        content.append({
            "type": "text",
            "text": prompt
        })

        if image_path:
            image_data = Image.open(image_path)
            content.append({
                "type": "image",
                "image": image_data
            })
        elif url:
            content.append({
                "type": "image",
                "url": url
            })
        else:
            return "無效的輸入，請提供圖片或 URL"

        conversation = [
            {
                "role": "user",
                "content": content
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Inference: Generation of the output
        output_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text

def main():
    default_prompt = "what is this"

    model = Qwen2VLForConditionalGeneration.from_pretrained(utils.get_vision_language_path())
    processor = AutoProcessor.from_pretrained(utils.get_vision_language_path())

    def image_request(prompt=default_prompt, image_path=None, url=None):
        content = []
        content.append({
            "type": "text",
            "text": prompt
        })

        if image_path:
            image_data = Image.open(image_path)
            content.append({
                "type": "image",
                "image": image_data
            })
        elif url:
            content.append({
                "type": "image",
                "url": url
            })
        else:
            return "無效的輸入，請提供圖片或 URL"

        conversation = [
            {
                "role":"user",
                "content":content
            }
        ]

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        # Inference: Generation of the output
        output_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text

    vlm = VisionLanguage()
    print(vlm.image_request(image_path="./star.jpg"))

if __name__ == '__main__':
    main()