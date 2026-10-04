from src.predict import predict

samples = [
    "This movie was amazing",
    "Worst film I have ever seen",
    "It was not bad",
    "I expected it to be great, but it wasn't",
    "Oh great, another boring sequel",
    "The acting was good but the story was terrible",
]

for s in samples:
    label, conf, _ = predict(s)
    print(f"{s!r} -> {label} ({conf:.1%})")