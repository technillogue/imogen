import ganclip_profiling as clipart

clipart.generate(
    clipart.base_args.with_update(
        {
            "text": "die demokratisches dance dance revolution deutsches Republik",
            "max_iterations": 100,
            "display_freq": 99,
        }
    )
)
