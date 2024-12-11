from random import random
import jiwer
import numpy as np
import torch
from datasets import load_dataset, Audio, Dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


if __name__ == "__main__":
    SPLIT = "validation"
    PAUSE = 1.0
    MAX_SINGLE_LAN_SEGMENT_LEN = 6.0

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    dataset_en = load_dataset("google/fleurs", "en_us")
    dataset_cs = load_dataset("google/fleurs", "cs_cz")
    dataset_en = dataset_en.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))
    dataset_cs = dataset_cs.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))

    gen_kwargs = {
        "max_new_tokens": 440,
        "num_beams": 1,
        "return_timestamps": True,
    }

    # Limit the samples length in the train split to 15s to be able to easily concatenate them
    for dataset in [dataset_en, dataset_cs]:
        dataset[SPLIT] = dataset[SPLIT].filter(
            lambda x: (x["num_samples"] / x['audio']['sampling_rate']) < MAX_SINGLE_LAN_SEGMENT_LEN)

    min_number_of_samples = min(len(dataset_en[SPLIT]), len(dataset_cs[SPLIT]))

    # create new dataset by concatenating samples from two languages
    gt = []
    hyp = []
    for index, (en_sample, cz_sample) in enumerate(zip(dataset_en[SPLIT], dataset_cs[SPLIT])):
        # select in-context example
        in_context_index = index
        while in_context_index == index:
            in_context_index = random.randint(0, min_number_of_samples)
        ic_en, ic_cz = dataset_en[SPLIT][in_context_index], dataset_cs[SPLIT][in_context_index]

        concatenated_sample = torch.from_numpy(np.concatenate(
            [
                # in-context example
                ic_en["audio"]["array"],
                np.zeros(round(ic_en["audio"]["sampling_rate"] * PAUSE)),
                ic_cz["audio"]["array"],
                np.zeros(round(ic_en["audio"]["sampling_rate"] * PAUSE)),
                # target utterance
                en_sample["audio"]["array"],
                np.zeros(round(en_sample["audio"]["sampling_rate"] * PAUSE)),
                cz_sample["audio"]["array"]
             ]))
        inputs = processor(concatenated_sample,
                           sampling_rate=en_sample["audio"]["sampling_rate"],
                           return_tensors="pt",
                           return_attention_mask=True).to(device, dtype=torch_dtype)

        # target utterance
        time_ic_en_end = round(ic_en['num_samples'] / 16_000, 2)
        time_ic_cz_start = time_ic_en_end + PAUSE
        time_ic_cz_end = time_ic_cz_start + round(ic_cz['num_samples'] / 16_000, 2)
        gt_context_readable = f"<|en|><|0.00|>{ic_en['transcription']}<|{time_ic_en_end}|> <|cs|><|{time_ic_cz_start}|>{ic_cz['transcription']}<|{time_ic_cz_end}|>"
        gt_context = f"<|en|>{ic_en['transcription']} <|cs|><|{time_ic_cz_start}|>{ic_cz['transcription']}<|{time_ic_cz_end}|>"

        time_en_start = time_ic_cz_end + PAUSE
        time_en_end = time_en_start + round(en_sample['num_samples'] / 16_000, 2)
        time_cz_start = time_en_end + PAUSE
        time_cz_end = time_cz_start + round(cz_sample['num_samples'] / 16_000, 2)
        # adding the timestamp for visual inspection only
        gt_text = f"<|en|><|{time_en_start}|>{en_sample['transcription']}<|{time_en_start}|> <|cs|><|{time_cz_start}|>{cz_sample['transcription']}<|{time_cz_end}|>"

        # set the prefix (in-context text) with forced_decoder_ids or input_ids
        # TODO:         forced_decoder_ids=[[0, 454], [2, 50360], [3, 50361]
            # map 0 to SOT
            # map 1 to <en>
            # map 2 to <transcribe>
            # 3 and onward is the transcript
            # TODO: not sure if we need to remove the lang tokens from the string
            # TODO: pycharm - inspect the tokenizer
        context_token_ids = processor.tokenizer.encode(gt_context, add_special_tokens=False)
        gen_kwargs.forced_decoder_ids = # TODO: see format above

        # TODO: make sure we don't duplicate <SOT>
        pred_ids = model.generate(**inputs, **gen_kwargs)
        generated_text = processor.decode(pred_ids[0], skip_special_tokens=True)
        gt_text_normalised = processor.tokenizer.decode(processor.tokenizer.encode(gt_text), skip_special_tokens=True)
        gt.append(gt_text_normalised)
        hyp.append(generated_text)
        print(processor.tokenizer.decode(processor.tokenizer.encode(gt_text), skip_special_tokens=True,
                                         decode_with_timestamps=True))
        print(processor.decode(pred_ids[0], skip_special_tokens=True, decode_with_timestamps=True))
        # TODO: do we need to remove the prefix from the decoded part?

    df = Dataset.from_dict({"gt": gt, "hyp": hyp})
    df.to_csv("results_icl.csv")
    metrics = jiwer.compute_measures(gt, hyp)
    print(metrics)