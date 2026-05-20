# PiMorph spatial pilot — AWS deploy

All commands run from your shell (`!aws ...` in this session). Credentials stay in `~/.aws/credentials`. Nothing secret enters the prompt.

## One-time setup

```
aws --profile redwing ec2 create-key-pair \
  --key-name pimorph-spatial \
  --query 'KeyMaterial' --output text > ~/.ssh/pimorph-spatial.pem
chmod 400 ~/.ssh/pimorph-spatial.pem

MYIP=$(curl -s https://checkip.amazonaws.com)/32
```

## Deploy stack

```
aws --profile redwing cloudformation deploy \
  --stack-name pimorph-spatial \
  --template-file infra/pimorph_spatial_stack.yaml \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides \
    KeyName=pimorph-spatial \
    AllowSshFromCidr=$MYIP \
    InstanceType=r6i.4xlarge \
    ScratchSizeGB=4000
```

## Get SSH / bucket info

```
aws --profile redwing cloudformation describe-stacks \
  --stack-name pimorph-spatial \
  --query "Stacks[0].Outputs" --output table
```

## Tear down (avoid storage charges)

```
aws --profile redwing s3 rm s3://pimorph-spatial-767397729725 --recursive
aws --profile redwing cloudformation delete-stack --stack-name pimorph-spatial
```

## Cost (us-east-1, on-demand)

| Resource | Rate | Monthly (24/7) |
|---|---|---|
| r6i.4xlarge | $1.008/hr | $735 |
| 4 TB gp3 (6k IOPS, 500 MB/s) | ~$340 |
| Bucket (2 TB std) | ~$46 |

Stop the instance when idle: `aws --profile redwing ec2 stop-instances --instance-ids <id>` — drops to ~$340/mo for EBS alone.
