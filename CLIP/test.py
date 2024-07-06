import argparse
import yaml

def run(args):
    print("Running training with the following settings:")
    # argsから直接属性にアクセスして表示
    print(f"batch_size: {args.batch_size}")
    print(f"epochs: {args.epochs}")
    print(f"lr: {args.lr}")
    print(f"device: {args.device}")
    print(f"num_workers: {args.num_workers}")
    print(f"seed: {args.seed}")
    print(f"use_wandb: {args.use_wandb}")
    print(f"data_dir: {args.data_dir}")
    print(f"model_path: {args.model_path}")
    print(f"embed_dim: {args.embed_dim}")
    print(f"image_resolution: {args.image_resolution}")
    print(f"vision_layers: {args.vision_layers}")
    print(f"vision_width: {args.vision_width}")
    print(f"vision_patch_size: {args.vision_patch_size}")
    print(f"context_length: {args.context_length}")
    print(f"vocab_size: {args.vocab_size}")
    print(f"transformer_width: {args.transformer_width}")
    print(f"transformer_heads: {args.transformer_heads}")
    print(f"transformer_layers: {args.transformer_layers}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the CLIP model training with specified configurations.')
    
    # コマンドライン引数を解析
    args = parser.parse_args()

    # YAML設定ファイルを読み込む
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        args.__dict__.update(config)  # argsの辞書を更新

    with open('configs/ViT.yaml', 'r') as f:
        vit_config = yaml.safe_load(f)
        vit_settings = vit_config['ViT-L/14-336px']
        args.__dict__.update(vit_settings)  # ViTの設定でargsを更新

    run(args)
