from training.train_segmentation import SegmentationTrainer, SegmentationDataset
import torch
from pathlib import Path


def main():
    print("=" * 50)
    print("  Video3D Segmentation Model Training")
    print("=" * 50)

    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_classes': 13,
        'in_channels': 6,
        'batch_size': 4,
        'num_epochs': 10,
        'learning_rate': 0.001
    }

    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print()

    dataset = SegmentationDataset(
        data_path='./data/datasets/S3DIS/processed',
        split='train',
        num_points=2048
    )

    print(f"Dataset size: {len(dataset)} samples")
    print()

    trainer = SegmentationTrainer(config=config)
    trainer.run(dataset, epochs=config['num_epochs'])

    checkpoint_dir = Path('data/checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / 'segmentation_model_final.pth'
    trainer.save_checkpoint(str(model_path))
    print(f"Model saved to: {model_path}")

    metrics = trainer.get_metrics()
    if metrics['train_loss']:
        print(f"Final train loss: {metrics['train_loss'][-1]:.4f}")
        print(f"Final train accuracy: {metrics['train_accuracy'][-1]:.2f}%")

    print()
    print("=" * 50)
    print("  Training finished!")
    print("=" * 50)


if __name__ == '__main__':
    main()
