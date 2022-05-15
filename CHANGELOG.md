## 1.3.5

- last time please 

## 1.3.4

- fix broken change from `reaction_predictor` to `likely_loss`

## 1.3.3

- more stupid fix

## 1.3.2

- hopefully fix inserted_ts

## 1.3.1

- only generate slug once and use that everywhere to keep the timestamp

## 1.3.0

- fuck it, good mode (aka likely) deserves a minor version bumo
- add timestamps to slug
- wait for seconds and try again before exiting on no prompt, to catch people asking for a prompt immediately after getting one back
- after getting an error, also unclaim the prompt (might turn out to be a bad idea) 
- also fix namespace lol

## 1.2.14

- good mode

## 1.2.13

- revert ViT/L

## 1.2.12.1

- clean up dockerfile

## 1.2.12

- ViT/L??

## 1.2.11

- feature flag for tweets

## 1.2.10 (unbuilt)

- check only for paid/same selector on scale in

## 1.2.9.1

- don't drop dot

## 1.2.9

- put just slug as filepath

## 1.2.8.1

- correct selector

## 1.2.8

- correct paid 

## 1.2.6

- fix

## 1.2.5

- torch = "^1.11.0" from pypi
- comments and stuff

## 1.2.4

- fix psycopg deps

## 1.2.3

- use correct content-type uploading to s3
- always use selector to avoid treating esrgan as a normal prompt
- poetry

## 1.2.2 

- upgrade pytorch version? apparent regression since 1.0.3

## 1.2.1

- fix scale in 

## 1.2

- use new selector field when getting prompts
- actually use scale in logic

## 1.1.2

- allow image prompts
- allow no text
- nopost param to not post to twitter
 
## 1.0.5

- store seed in DB and retry uploads

## 1.0.3 and .4

- emoji admin messages
- cuda 113 for a6000 support
- free workers don't process paid
- 
- ...
