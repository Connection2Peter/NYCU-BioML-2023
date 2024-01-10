from lib import transformer



model = transformer.TFM(
    maxlen=21*60,
    num_heads=2,
    ff_dim=128,
    embed_dim=128,
    vocab_size=10001,
    num_class=2,
)
model.summary()