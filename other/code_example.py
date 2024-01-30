sys_prompt = "I am OrcaPhi. The following is my internal dialogue as an AI assistant.\n" \
        "Today is September 15, 2023. I have no access to outside tools, news, or current events.\n" \
        "I carefully provide accurate, factual, thoughtful, nuanced answers and am brilliant at reasoning.\n" \
        "I think through my answers step-by-step to be sure I always get the right answer.\n" \
        "I think more clearly if I write out my thought process in a scratchpad manner first; therefore, I always " \
        "explain background context, assumptions, and step-by-step thinking BEFORE trying to answer a question." \
        "Take a deep breath and think calmly about everything presented."

    prefix = "<|im_start|>"
    suffix = "<|im_end|>\n"
    sys_format = prefix + "system\n" + sys_prompt + suffix


def my_pred(model, tokenizer, prompt, img, n=20, alpha=1):
    user_format = prefix + "user\n" + prompt + suffix
    assistant_format = prefix + "assistant\n"
    input_text = sys_format + user_format + assistant_format
    input_ids = tokenizer(input_text)['input_ids']
    for i in range(n):
        input_tns = torch.tensor(input_ids).unsqueeze(0)
        out = model(input_ids=input_tns.cuda(), images=img.cuda(), alpha=alpha)
        new_id = out.logits.argmax(-1)[-1, -1]
        input_ids.append(new_id.item())
    print(tokenizer.decode(input_ids[143:]))

def my_pred_unravel():
    user_format = prefix + "user\n" + prompt + suffix
    assistant_format = prefix + "assistant\n"
    input_text = sys_format + user_format + assistant_format
    tkn = tokenizer1(input_text)
    input_ids = tkn.input_ids
    for i in range(20):
        inputs_embeds = None
        if inputs_embeds is None:
            inputs_embeds = model1.model.layers[0](torch.tensor(input_ids).cuda())
        hidden_layer = inputs_embeds
        for module in model1.model.layers[1:-1]:
            hidden_layer = module(hidden_layer, past_cache=None)
        lm_logits = model1.model.layers[-1](hidden_layer)
        new_t = lm_logits.argmax(-1)[-1, -1].item()
        input_ids.append(new_t)
    print(tokenizer1.decode(input_ids))
