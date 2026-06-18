import os
import sys
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline

def test_lora_inference(lora_path, prompt, output_path, model_name="CompVis/stable-diffusion-v1-4"):
    """測試LoRA推理"""
    try:
        print(f"[INFO] 載入模型: {model_name}")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            safety_checker=None,  # 禁用安全檢查加快載入
            requires_safety_checker=False
        )
        
        if lora_path and os.path.exists(lora_path):
            print(f"[INFO] 載入LoRA權重: {lora_path}")
            pipe.load_lora_weights(lora_path)
        else:
            print(f"[WARN] LoRA路徑不存在，使用原始模型: {lora_path}")
        
        pipe = pipe.to("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"[INFO] 生成圖像，提示: {prompt}")
        torch.manual_seed(42)  # 固定隨機種子
        
        image = pipe(
            prompt, 
            num_inference_steps=20,  # 減少步數加快生成
            guidaFnce_scale=7.5,
            height=512,
            width=512
        ).images[0]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"[INFO] 圖像已保存: {output_path}")
        
        return True
    except Exception as e:
        print(f"[ERROR] 推理失敗: {str(e)}")
        return False

def main():
    # 測試配置
    test_configs = [
        {
            "name": "Artistic Custom LoRA", 
            "lora_path": "./runs/artistic_custom",
            "prompt": "a house in artistic style",
            "output_path": "./outputs/artistic_custom_test.png"
        },
        {
            "name": "DreamBooth Cat LoRA",
            "lora_path": "./runs/dreambooth_cat", 
            "prompt": "a photo of sks cat sitting on a chair",
            "output_path": "./outputs/dreambooth_cat_test.png"
        },
        {
            "name": "Original Model (No LoRA)",
            "lora_path": None,
            "prompt": "a beautiful landscape painting", 
            "output_path": "./outputs/original_model_test.png"
        }
    ]
    
    successful_tests = 0
    total_tests = len(test_configs)
    
    for config in test_configs:
        print(f"\n{'='*50}")
        print(f"測試: {config['name']}")
        print(f"{'='*50}")
        
        if test_lora_inference(
            config["lora_path"],
            config["prompt"], 
            config["output_path"]
        ):
            successful_tests += 1
            print(f"✅ {config['name']} 測試成功")
        else:
            print(f"❌ {config['name']} 測試失敗")
    
    print(f"\n推理測試完成: {successful_tests}/{total_tests} 成功")
    return successful_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
