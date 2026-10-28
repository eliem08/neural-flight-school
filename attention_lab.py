"""Build an educational, randomly initialized tiny transformer and visualize attention.

The model is intentionally untrained: attention plots teach tensor mechanics only and
must not be interpreted as learned language behavior.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch import nn
import matplotlib.pyplot as plt
from provenance import build_manifest

class TinyAttentionModel(nn.Module):
    def __init__(self, vocab_size=128, width=64, heads=4, layers=2, max_length=128):
        super().__init__()
        if width % heads: raise ValueError("width must be divisible by heads")
        self.embedding=nn.Embedding(vocab_size,width);self.position=nn.Embedding(max_length,width)
        self.blocks=nn.ModuleList([nn.MultiheadAttention(width,heads,batch_first=True) for _ in range(layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.head=nn.Linear(width,vocab_size);self.heads=heads
    def forward(self, ids):
        b,t=ids.shape
        x=self.embedding(ids)+self.position(torch.arange(t,device=ids.device))[None,:,:]
        causal=torch.triu(torch.ones(t,t,device=ids.device,dtype=torch.bool),diagonal=1)
        attentions=[]
        for attn,norm in zip(self.blocks,self.norms):
            update,weights=attn(x,x,x,attn_mask=causal,need_weights=True,average_attn_weights=False)
            x=norm(x+update);attentions.append(weights)
        return self.head(x),attentions

def run(text="attention is a computation", output="outputs/attention-lab", seed=7, device="auto"):
    torch.manual_seed(seed)
    device=device if device!="auto" else ("cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cpu")
    # Byte-level IDs keep the demo dependency-free and reproducible.
    ids=torch.tensor([[b % 128 for b in text.encode("utf-8")[:128]] or [1]],device=device)
    model=TinyAttentionModel(max_length=max(128,ids.shape[1])).to(device).eval()
    with torch.inference_mode(): logits,attentions=model(ids)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    tokens=[f"{i}:{chr(b) if 32<=b<127 else hex(b)}" for i,b in enumerate(text.encode("utf-8")[:ids.shape[1]])]
    summaries=[]
    for li,weights in enumerate(attentions):
        matrix=weights[0].float().mean(dim=0).cpu().numpy()
        fig,ax=plt.subplots(figsize=(max(6,min(14,ids.shape[1]*0.45)),max(4,min(12,ids.shape[1]*0.4))))
        im=ax.imshow(matrix,aspect="auto",origin="upper");ax.set_title(f"Untrained causal attention — layer {li}");ax.set_xlabel("Key position");ax.set_ylabel("Query position");fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/f"attention-layer-{li}.png",dpi=150);plt.close(fig)
        summaries.append({"layer":li,"shape":list(weights.shape),"mean_attention_entropy":float((-(weights.float().clamp_min(1e-9)*weights.float().clamp_min(1e-9).log()).sum(dim=-1)).mean().cpu())})
    report={"device":device,"seed":seed,"input_text":text,"token_count":int(ids.shape[1]),"logits_shape":list(logits.shape),"layers":summaries,"manifest":build_manifest("neural-flight-school",device=device,seed=seed,config={"vocab_size":128,"width":64,"heads":4,"layers":2},inputs={"text":text}),"warning":"Random weights are untrained; visualizations are educational, not evidence about a trained model."}
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--text",default="attention is a computation");p.add_argument("--output",default="outputs/attention-lab");p.add_argument("--seed",type=int,default=7);p.add_argument("--device",choices=["auto","cpu","cuda","mps"],default="auto")
    a=p.parse_args();print(json.dumps(run(a.text,a.output,a.seed,a.device),indent=2))
if __name__=="__main__":main()
