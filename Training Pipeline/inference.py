def image_inference(model, src, device, show=False, save=False):
    results = model.predict(source=src, show=show, device=device, save=save)
    process_results(results[0])


def stream_inference(model, src, device, show=False, save=False):
    for result in model.predict(source=src, show=show, device=device, save=save, stream=True, verbose=False):
        pass
