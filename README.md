```bash
./install_cuda.sh # to get nvidia-smi on stock ubuntu
./install_deps.sh # or pipenv + git clone
./download_modals.sh
./postgres_jobs.py
```

secrets:

* `DATABASE_URL`: postgres db with the prompt queue
* `SUPABASE_API_KEY`: supabase api key for uploading images 
* `REDIS_URL`: redis url for downloading initial/target images

flags:

* `POWEROFF`: poweroff when queue empty
* `EXIT`: exit when queue empty
* `EXIT_ON_LOAD`: exit after finishing a prompt if paid_queue_size / workers > 5 (and not last worker)
* `FREE`: handle unpaid prompts (only handle paid prompts by default)
* `SELECTOR`: only handle prompts with the matching selector 
