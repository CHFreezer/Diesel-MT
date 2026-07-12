# Simple Encoder-Decoder parameter estimator.
# Assumes shared encoder embedding, decoder embedding, and output head weights.

CONFIGS = [
    {
        "name": "target_200m",
        "vocab_size": 65_536,
        "d_model": 768,
        "ffn_dim": 3_072,
        "encoder_layers": 16,
        "decoder_layers": 4,
        "attention_heads": 12,
    },
    {
        "name": "mvp_e12_d3_v48k",
        "vocab_size": 49_152,
        "d_model": 512,
        "ffn_dim": 2_048,
        "encoder_layers": 12,
        "decoder_layers": 3,
        "attention_heads": 8,
    },
    {
        "name": "mvp_e12_d3_v32k",
        "vocab_size": 32_768,
        "d_model": 512,
        "ffn_dim": 2_048,
        "encoder_layers": 12,
        "decoder_layers": 3,
        "attention_heads": 8,
    },
    {
        "name": "mvp_e8_d2_v48k",
        "vocab_size": 49_152,
        "d_model": 512,
        "ffn_dim": 2_048,
        "encoder_layers": 8,
        "decoder_layers": 2,
        "attention_heads": 8,
    },
    {
        "name": "mvp_e8_d2_v32k",
        "vocab_size": 32_768,
        "d_model": 512,
        "ffn_dim": 2_048,
        "encoder_layers": 8,
        "decoder_layers": 2,
        "attention_heads": 8,
    },
]


def estimate(config: dict[str, int | str]) -> dict[str, float | int | str]:
    d_model = int(config["d_model"])
    ffn_dim = int(config["ffn_dim"])
    vocab_size = int(config["vocab_size"])
    encoder_layers = int(config["encoder_layers"])
    decoder_layers = int(config["decoder_layers"])

    embedding = vocab_size * d_model
    encoder_per_layer = 4 * d_model**2 + 2 * d_model * ffn_dim
    decoder_per_layer = 8 * d_model**2 + 2 * d_model * ffn_dim
    encoder = encoder_per_layer * encoder_layers
    decoder = decoder_per_layer * decoder_layers
    total = embedding + encoder + decoder

    return {
        **config,
        "embedding": embedding,
        "encoder_per_layer": encoder_per_layer,
        "decoder_per_layer": decoder_per_layer,
        "encoder": encoder,
        "decoder": decoder,
        "total": total,
        "fp16_mib": total * 2 / 1024**2,
        "int8_mib": total / 1024**2,
        "int4_mib": total / 2 / 1024**2,
    }


def print_report(result: dict[str, float | int | str]) -> None:
    total = int(result["total"])
    embedding = int(result["embedding"])
    encoder = int(result["encoder"])
    decoder = int(result["decoder"])

    print(f"\n[{result['name']}]")
    print(
        "config: "
        f"vocab={result['vocab_size']:,}, "
        f"d_model={result['d_model']}, "
        f"ffn={result['ffn_dim']}, "
        f"encoder={result['encoder_layers']}, "
        f"decoder={result['decoder_layers']}, "
        f"heads={result['attention_heads']}"
    )
    print(f"Tokenizer/Embedding params: {embedding / 1e6:.2f}M")
    print(f"Encoder params: {encoder / 1e6:.2f}M")
    print(f"Decoder params: {decoder / 1e6:.2f}M")
    print(f"Total params: {total / 1e6:.2f}M")
    print(f"BF16/FP16 weights: {float(result['fp16_mib']):.1f} MiB")
    print(f"INT8 weights: {float(result['int8_mib']):.1f} MiB")
    print(f"INT4 theoretical weights: {float(result['int4_mib']):.1f} MiB")


if __name__ == "__main__":
    for cfg in CONFIGS:
        print_report(estimate(cfg))
