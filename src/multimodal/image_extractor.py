"""
Multimodal Processing - Image Extraction from PDFs
Quick implementation to demonstrate multimodal capabilities
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict
from PIL import Image
import io
import base64

class ImageExtractor:
    """Extract images from research papers"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("data/images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_images_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Extract all images from a PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of image info dictionaries
        """
        print(f"📸 Extracting images from: {pdf_path.name}")
        
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Save image
                    image_filename = f"{pdf_path.stem}_page{page_num+1}_img{img_index+1}.{image_ext}"
                    image_path = self.output_dir / image_filename
                    
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    # Get image dimensions
                    img_obj = Image.open(io.BytesIO(image_bytes))
                    width, height = img_obj.size
                    
                    images.append({
                        'filename': image_filename,
                        'path': str(image_path),
                        'page': page_num + 1,
                        'index': img_index + 1,
                        'format': image_ext,
                        'width': width,
                        'height': height,
                        'size_kb': len(image_bytes) / 1024
                    })
                    
                except Exception as e:
                    print(f"  ⚠️ Error extracting image: {str(e)}")
        
        doc.close()
        
        print(f"  ✅ Extracted {len(images)} images")
        return images
    
    def extract_from_directory(self, papers_dir: Path) -> Dict:
        """Extract images from all PDFs in directory"""
        pdf_files = list(papers_dir.glob("*.pdf"))
        
        all_images = {}
        total_images = 0
        
        for pdf_path in pdf_files:
            images = self.extract_images_from_pdf(pdf_path)
            all_images[pdf_path.name] = images
            total_images += len(images)
        
        print(f"\n✅ Total: {total_images} images from {len(pdf_files)} papers")
        return all_images
    
    def get_image_summary(self, images: List[Dict]) -> Dict:
        """Get summary statistics of extracted images"""
        if not images:
            return {}
        
        total_size = sum(img['size_kb'] for img in images)
        formats = {}
        
        for img in images:
            fmt = img['format']
            formats[fmt] = formats.get(fmt, 0) + 1
        
        return {
            'total_images': len(images),
            'total_size_mb': total_size / 1024,
            'formats': formats,
            'avg_size_kb': total_size / len(images)
        }
    
    def create_image_index(self, all_images: Dict) -> str:
        """Create a markdown index of all images"""
        md_content = ["# Extracted Images Index\n"]
        
        for pdf_name, images in all_images.items():
            md_content.append(f"\n## {pdf_name}\n")
            md_content.append(f"Total images: {len(images)}\n")
            
            for img in images:
                md_content.append(f"- **Page {img['page']}, Image {img['index']}**: "
                               f"{img['format'].upper()} | {img['width']}x{img['height']} | "
                               f"{img['size_kb']:.1f} KB\n")
                md_content.append(f"  - File: `{img['filename']}`\n")
        
        index_path = self.output_dir / "image_index.md"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.writelines(md_content)
        
        return str(index_path)

class TableExtractor:
    """Simple table detection and extraction"""
    
    def __init__(self):
        pass
    
    def detect_tables(self, pdf_path: Path) -> List[Dict]:
        """
        Detect potential tables in PDF (simple heuristic approach)
        
        For full implementation, would use Camelot or Tabula
        This is a simplified version for demonstration
        """
        print(f"📊 Detecting tables in: {pdf_path.name}")
        
        doc = fitz.open(pdf_path)
        tables = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Simple heuristic: look for text with many tabs or aligned columns
            lines = text.split('\n')
            potential_table_lines = []
            
            for line in lines:
                # If line has multiple tabs or spaces suggesting columns
                if '\t' in line or '  ' in line:
                    potential_table_lines.append(line)
            
            if len(potential_table_lines) > 3:  # At least 3 rows
                tables.append({
                    'page': page_num + 1,
                    'estimated_rows': len(potential_table_lines),
                    'preview': '\n'.join(potential_table_lines[:3])
                })
        
        doc.close()
        
        print(f"  ✅ Found {len(tables)} potential tables")
        return tables

def demo_multimodal():
    """Demo multimodal features"""
    print("=" * 70)
    print("🖼️  Multimodal Processing Demo")
    print("=" * 70)
    
    # Initialize extractors
    image_extractor = ImageExtractor()
    table_extractor = TableExtractor()
    
    # Extract images
    papers_dir = Path("data/papers/raw")
    
    if papers_dir.exists():
        # Extract images
        all_images = image_extractor.extract_from_directory(papers_dir)
        
        # Create index
        index_path = image_extractor.create_image_index(all_images)
        print(f"\n📝 Image index created: {index_path}")
        
        # Get summary
        all_imgs_list = [img for imgs in all_images.values() for img in imgs]
        summary = image_extractor.get_image_summary(all_imgs_list)
        
        print(f"\n📊 Summary:")
        print(f"  Total images: {summary.get('total_images', 0)}")
        print(f"  Total size: {summary.get('total_size_mb', 0):.2f} MB")
        print(f"  Formats: {summary.get('formats', {})}")
        
        # Detect tables (in first PDF)
        pdf_files = list(papers_dir.glob("*.pdf"))
        if pdf_files:
            tables = table_extractor.detect_tables(pdf_files[0])
            print(f"\n📊 Detected {len(tables)} potential tables in {pdf_files[0].name}")
    else:
        print(f"❌ Papers directory not found: {papers_dir}")

if __name__ == "__main__":
    demo_multimodal()