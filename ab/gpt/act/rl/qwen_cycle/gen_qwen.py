import json, sys, time
def main():
    from transformers import AutoTokenizer
    import ab.gpt.util.llm.VLLMSleep as V
    MODEL = __import__("os").environ.get("MODEL", "Qwen/Qwen3.5-9B"); tok = AutoTokenizer.from_pretrained(MODEL)
    eng = V.build_engine(MODEL, tok, 8192, temperature=0.8, top_p=0.9, top_k=50)
    eng.suppress_thinking = True
    for pf, out in zip(sys.argv[1::2], sys.argv[2::2]):
        P = json.load(open(pf)); t = time.time()
        outs = eng.chat_batch(P, max_new_tokens=int(__import__("os").environ.get("MAX_NEW", "2048")))
        with open(out, "w") as f:
            for i, (p, o) in enumerate(zip(P, outs)):
                text = o if isinstance(o, str) else next(x for x in o if isinstance(x, str))
                f.write(json.dumps(dict(idx=i, prompt=p, text=text, n_tokens=len(tok.encode(text)))) + "\n")
        print(f"{pf}: {len(outs)} answers in {time.time()-t:.0f}s -> {out}", flush=True)
if __name__ == "__main__":
    main()
