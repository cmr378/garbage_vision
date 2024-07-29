import os, argparse
from duckduckgo_search import DDGS
from fastcore.all import *
from fastbook import *
from pathlib import Path


class ImageProcessor:
    def __init__(self):
        self.dataloaders = None

    def main(self):
        parser = argparse.ArgumentParser(description="Image processing script")
        parser.add_argument('--download', action='store_true', help='Download images')
        parser.add_argument('--dataloaders', action='store_true', help='Create dataloaders')
        parser.add_argument('--train', action='store_true', help='Train the model')

        args = parser.parse_args()

        if args.download:
            self.download_images_ddg(
                path=Path('images'), 
                img_types=('plastic litter on the city street: litter only', 'cigarette butts'), 
                img_category='litter',
                num_images=100
            )
        else:
            print('Download flag not set, not downloading images from DDG')
 
        if args.dataloaders:
            self.dataloaders = self.create_dataloaders(
                path=Path('images')
            )
        else:
            print('Dataloaders flag not set, skipping dataloader creation')

        if args.train:
            print('dataloaders',self.dataloaders)
            self.train_model(
                path=Path('images'),
                dataloaders=self.dataloaders
            )
        else:
            print('Train flag not set, skipping training')

    def download_images_ddg(self, path: Path, img_category: str, img_types: tuple, num_images: int):
        print('Downloading images')
        if not path.exists():
            path.mkdir(parents=True)
            
        for o in img_types:
            dest = path / o.replace(" ", "_")
            dest.mkdir(parents=True, exist_ok=True)
            search_query = f'{o} {img_category}'
            
            results = search_images_ddg(search_query, max_results=num_images)
            
            print(f'Found {len(results)} results for "{search_query}"')
            
            for idx, result in enumerate(results):
                try:
                    download_url(result['image'], dest/f'{o.replace(" ", "_")}-{idx+1}.jpg', show_progress=False)
                except Exception as e:
                    print(f'Error downloading image {idx+1}: {e}')

    def create_dataloaders(self, path: str) -> Dict[str, DataLoaders]:
        
        path = Path(path)
        fns = get_image_files(path)
        print(f"Found {len(fns)} image files")
        failed = verify_images(fns)
        print(f"Found {len(failed)} failed image files")
        failed.map(Path.unlink)
        
        dataloaders = {}

        for item in path.iterdir():
            if item.is_dir():
                print('Processing', item)
                # List files in the directory for debugging
                item_files = get_image_files(item)
                print(f"Found {len(item_files)} files in {item}")
                litter_type_datablock = DataBlock(
                    blocks=(ImageBlock, CategoryBlock), 
                    get_items=get_image_files, 
                    splitter=RandomSplitter(valid_pct=0.2, seed=42),
                    get_y=parent_label,
                    item_tfms=Resize(128)
                )
                dls = litter_type_datablock.dataloaders(item)
                dataloaders[item.name] = dls
                dls.valid.show_batch(max_n=4, nrows=1)
        return dataloaders

    def train_model(self, path: str, dataloaders: Dict[str, DataLoaders]):
        # Placeholder for training logic
        print('Training model with provided dataloaders')
        for k,v in dataloaders.items():
            print(k,v,path)
            learn = vision_learner(v, resnet18, metrics=error_rate)
            learn.fine_tune(4)



if __name__ == "__main__":
    processor = ImageProcessor()
    processor.main()