# model inference
# Ensure that the model path can be imported correctly
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def model_inference(model_components, image_path_lst, qs_lst, args):
    # You can modify your model inference functions here. There are some examples as follows:
    # 1. TinyLLaVA
    if args.model_name in ['TinyLLaVA-Phi-2-SigLIP-3.1B', 'TinyLLaVA-Qwen2-0.5B-SigLIP', 'TinyLLaVA-Qwen2.5-3B-SigLIP']:
        from models.TinyLLaVA_Factory.tinyllava_model import tinyllava_inference
        model = model_components.get("model")
        tokenizer = model_components.get("tokenizer")
        text_processor = model_components.get("text_processor")
        image_processor = model_components.get("image_processor")
        output_lst = tinyllava_inference(image_path_lst, qs_lst, image_processor, text_processor, tokenizer, model, args)

    # 2. TinyLLaVA for SFT
    elif args.model_name in ['TinyLLaVA-Qwen2-0.5B-SigLIP-Baseline', 'TinyLLaVA-Qwen2-0.5B-SigLIP-QA',  \
                             'TinyLLaVA-Qwen2-0.5B-SigLIP-GRT', 'TinyLLaVA-Qwen2.5-3B-SigLIP-GRT', \
                             'TinyLLaVA-Qwen2-0.5B-SigLIP-GRT-HELPER', 'TinyLLaVA-Qwen2-0.5B-SigLIP-GRT-CAT']:
        from models.TinyLLaVA_Factory_SFT.tinyllava_sft_model import tinyllava_sft_inference
        model = model_components.get("model")
        tokenizer = model_components.get("tokenizer")
        text_processor = model_components.get("text_processor")
        image_processor = model_components.get("image_processor")        
        output_lst = tinyllava_uni_inference(image_path_lst, qs_lst, image_processor, text_processor, tokenizer, model, args)

    # 3.InternVL        
    elif args.model_name in ['InternVL2_5-1B']:
        from models.InternVL.internvl_model import internvl_inference
        model = model_components.get("model")
        tokenizer = model_components.get("tokenizer")
        generation_config = model_components.get("generation_config")
        output_lst = internvl_inference(image_path_lst, qs_lst, model, tokenizer, generation_config, args)

    # 4. LLaVA
    elif args.model_name in ['llava-v1.5-7b']:
        from models.LLaVA.llava_model import llava_inference
        model = model_components.get("model")
        tokenizer = model_components.get("tokenizer")
        context_len = model_components.get("context_len")
        image_processor = model_components.get("image_processor")
        output_lst = llava_inference(image_path_lst, qs_lst, image_processor, context_len, tokenizer, model, args)

    # 5. LLaVA for SFT/ llava-hf
    elif args.model_name in ['llava-1.5-7b-hf', 'llava-v1.5-7b-Baseline', 'llava-v1.5-7b-QA', 'llava-v1.5-7b-GRT']:
        from models.LLaVA.llava_hf_model import llava_hf_inference
        model = model_components.get("model")
        processor = model_components.get("processor")       
        output_lst = llava_hf_inference(image_path_lst, qs_lst, model, processor, args)

    # 6. Qwen2.5-VL
    elif args.model_name in ['Qwen2.5-VL-3B-Instruct']:
        from models.Qwen.qwen_model import qwen_inference
        model = model_components.get("model")
        processor = model_components.get("processor")
        output_lst = qwen_inference(image_path_lst, qs_lst, model, processor, args)

    # 7. Qwen2-VL / Qwen2-VL for SFT
    elif args.model_name in ['Qwen2-VL-2B', 'Qwen2-VL-2B-Baseline', 'Qwen2-VL-2B-QA', 'Qwen2-VL-2B-GRT']:
        from models.Qwen.qwen2_model import qwen2_inference
        model = model_components.get("model")
        processor = model_components.get("processor")
        output_lst = qwen2_inference(image_path_lst, qs_lst, model, processor, args)

    # 8. deepseek
    elif args.model_name in ['deepseek-vl2-tiny']:
        from models.DeepSeek_VL2.deepseek_model import deepseek_inference
        model = model_components.get("model")
        tokenizer = model_components.get("tokenizer")
        vl_chat_processor = model_components.get("vl_chat_processor")
        output_lst = deepseek_inference(image_path_lst, qs_lst, vl_chat_processor, tokenizer, model, args)

    # 9. BLIP-3
    elif args.model_name in ['xgen-mm-phi3-mini-instruct-interleave-r-v1.5']:
        from models.LAVIS_xgen_mm.blip3_model import blip3_inference
        model = model_components.get("model")
        tokenizer = model_components.get("tokenizer")
        image_processor = model_components.get("image_processor")
        output_lst = blip3_inference(image_path_lst, qs_lst, model, tokenizer, image_processor, args)
        
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")
    
    return output_lst