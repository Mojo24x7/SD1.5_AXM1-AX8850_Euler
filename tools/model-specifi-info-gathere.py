import os, json
import axengine

base="axmodels"
models={
  "text_encoder": f"{base}/sd15_text_encoder_sim.axmodel",
  "unet": f"{base}/unet.axmodel",
  "vae_decoder": f"{base}/vae_decoder.axmodel",
  "vae_encoder": f"{base}/vae_encoder.axmodel",
}

def show_io(name, path):
    print("\n===", name, "===")
    s=axengine.InferenceSession(path, providers=["AXCLRTExecutionProvider"])
    ins=getattr(s,"get_inputs",lambda:[])()
    outs=getattr(s,"get_outputs",lambda:[])()
    print("inputs:")
    for i,x in enumerate(ins):
        # try common attributes
        n=getattr(x,"name",None)
        shp=getattr(x,"shape",None)
        dt=getattr(x,"dtype",None)
        print(f"  {i}: name={n} shape={shp} dtype={dt}")
    print("outputs:")
    for i,x in enumerate(outs):
        n=getattr(x,"name",None)
        shp=getattr(x,"shape",None)
        dt=getattr(x,"dtype",None)
        print(f"  {i}: name={n} shape={shp} dtype={dt}")
    del s

for k,p in models.items():
    if os.path.exists(p):
        show_io(k,p)
    else:
        print("MISSING:", p)
