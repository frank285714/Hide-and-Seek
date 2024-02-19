from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from demo_model import hide_model_path, base_model_dir, bnb_config


# def test_hide_model():
#     base_model = 'bloomz-560m'
#     # base_model = 'bloomz-1b7'
#     base_model_dir = f'./models/{base_model}'
#     model = AutoModelForCausalLM.from_pretrained(base_model_dir, load_in_4bit=True, quantization_config=bnb_config,
#                                                  # device_map='cuda:0',
#                                                  trust_remote_code=True, torch_dtype=torch.float16)
#     hide_model = PeftModel.from_pretrained(model, hide_model_path, quantization_config=bnb_config, device_map='cuda:0',
#                                            trust_remote_code=True)

class LLMTest:

    def __init__(self,is_remote=False):
        # Load model directly
        if is_remote:
            _path = "bigscience/bloom-560m"
        else:
            _path = r"E:\Documents\Codes\PycharmProjects\Hide-and_Seek\models\bloomz-560m"
        # from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(_path, trust_remote_code=is_remote)
        self.model = AutoModelForCausalLM.from_pretrained(_path, trust_remote_code=is_remote)

    def generate(self,input_text='你是谁'):
        inputs = self.tokenizer(input_text, return_tensors='pt')
        pred = self.model.generate(**inputs, generation_config=GenerationConfig(
            max_new_tokens=int(len(inputs['input_ids'][0]) * 3),
            do_sample=False,
            num_beams=3,
            repetition_penalty=5.0,
        ))
        pred = pred.cpu()[0][len(inputs['input_ids'][0]):]
        response = self.tokenizer.decode(pred, skip_special_tokens=True).split('\n')[0]
        # torch.cuda.empty_cache()
        # gc.collect()
        return response



if __name__ == '__main__':
    model=LLMTest()
    txt='请提取下文中的人名、时间、地理信息，并以json格式输出：\n2023年5月1日，马云和王小磊在杭州创办了一家新的科技公司。'
    resp=model.generate(txt)
