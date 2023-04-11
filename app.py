from diffusers import StableDiffusionPipeline
import torch


if __name__ == '__main__':
    model_id = "CompVis/stable-diffusion-v1-1"
    # model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt = input("Enter your prompt here: ")
    image = pipe(prompt).images[0]  
        
    image.save("image.png")
