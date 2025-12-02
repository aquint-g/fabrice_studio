try:
    import vertexai.preview.vision_models
    print("Available in vertexai.preview.vision_models:", dir(vertexai.preview.vision_models))
except ImportError as e:
    print("ImportError:", e)

try:
    from vertexai.preview.generative_models import Part
    print("Part imported successfully")
except ImportError as e:
    print("Part ImportError:", e)
