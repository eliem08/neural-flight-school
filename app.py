import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import GPT2Config,GPT2LMHeadModel
from runtime import device_name,OUT
from provenance import build_manifest

def run(text=None,real_model=False,model_id=None):
 text=text or "Neurons learn patterns"; device=device_name(); chars=sorted(set(text)); vocab={c:i+1 for i,c in enumerate(chars)}
 ids=torch.tensor([[vocab[c] for c in text]],dtype=torch.long,device=device)
 config=GPT2Config(vocab_size=max(vocab.values())+1,n_positions=max(64,len(text)+8),n_ctx=max(64,len(text)+8),n_embd=32,n_layer=2,n_head=2)
 torch.manual_seed(19); model=GPT2LMHeadModel(config).to(device).eval()
 with torch.no_grad(): out=model(input_ids=ids,output_hidden_states=True,output_attentions=True,return_dict=True)
 hidden=out.hidden_states[-1][0].detach().float().cpu().T
 fig,ax=plt.subplots(figsize=(10,4)); ax.imshow(hidden.numpy(),aspect="auto",interpolation="nearest"); ax.set(title="Tiny transformer activation map",xlabel="character position",ylabel="hidden feature"); fig.tight_layout()
 plot=OUT/"flight_school_activations.png"; fig.savefig(plot,dpi=130); plt.close(fig)
 report={"model":"tiny-random-gpt2","device":device,"text":text,"characters":list(text),"hidden_state_shapes":[list(x.shape) for x in out.hidden_states],"attention_shapes":[list(x.shape) for x in out.attentions],"plot":str(plot),"caveat":"Random weights demonstrate tensor flow only; they do not encode trained language knowledge.","manifest":build_manifest("neural-flight-school",model_id="tiny-random-gpt2",device=device,seed=19,config={"architecture":"GPT2","n_embd":32,"n_layer":2,"n_head":2,"do_sample":False},inputs={"text":text})}
 (OUT/"flight_school.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); return report
