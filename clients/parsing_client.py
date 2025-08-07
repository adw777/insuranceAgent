import pymupdf4llm
import pathlib
import os
from typing import Optional, List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFParsingClient:
    """Client for parsing PDF files to markdown format using pymupdf4llm."""
    
    def __init__(self):
        """Initialize the PDF parsing client."""
        pass
    
    def parse_single_pdf(self, pdf_path: str) -> str:
        """
        Parse a single PDF file and return the parsed text as markdown.
        
        Args:
            pdf_path (str): Path to the PDF file to parse
            
        Returns:
            str: Parsed text in markdown format
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            Exception: If parsing fails
        """
        try:
            # Check if file exists
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Check if file is a PDF
            if not pdf_path.lower().endswith('.pdf'):
                raise ValueError(f"File must be a PDF: {pdf_path}")
            
            logger.info(f"Parsing PDF: {pdf_path}")
            
            # Parse PDF to markdown
            md_text = pymupdf4llm.to_markdown(pdf_path)
            
            logger.info(f"Successfully parsed PDF: {pdf_path}")
            return md_text
            
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {str(e)}")
            raise
    
    def parse_pdf_folder(self, 
                        input_folder: str, 
                        output_folder: str = "parsed_markdown",
                        overwrite: bool = False) -> Dict[str, str]:
        """
        Parse all PDF files from a folder and save markdown files to output folder.
        
        Args:
            input_folder (str): Path to folder containing PDF files
            output_folder (str): Path to folder where markdown files will be saved
            overwrite (bool): Whether to overwrite existing markdown files
            
        Returns:
            Dict[str, str]: Dictionary mapping PDF filenames to their status
                           (success/error message)
        """
        results = {}
        
        try:
            # Check if input folder exists
            if not os.path.exists(input_folder):
                raise FileNotFoundError(f"Input folder not found: {input_folder}")
            
            # Create output folder if it doesn't exist
            pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)
            logger.info(f"Output folder created/verified: {output_folder}")
            
            # Get all PDF files from input folder
            pdf_files = [f for f in os.listdir(input_folder) 
                        if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                logger.warning(f"No PDF files found in {input_folder}")
                return {"warning": "No PDF files found in the specified folder"}
            
            logger.info(f"Found {len(pdf_files)} PDF files to process")
            
            # Process each PDF file
            for pdf_file in pdf_files:
                pdf_path = os.path.join(input_folder, pdf_file)
                
                # Generate output filename (replace .pdf with .md)
                md_filename = os.path.splitext(pdf_file)[0] + ".md"
                md_path = os.path.join(output_folder, md_filename)
                
                try:
                    # Check if output file already exists
                    if os.path.exists(md_path) and not overwrite:
                        results[pdf_file] = f"Skipped - file already exists: {md_filename}"
                        logger.info(f"Skipped {pdf_file} - output file already exists")
                        continue
                    
                    # Parse the PDF
                    md_text = self.parse_single_pdf(pdf_path)
                    
                    # Save markdown to output folder
                    pathlib.Path(md_path).write_text(md_text, encoding='utf-8')
                    
                    results[pdf_file] = f"Success - saved as: {md_filename}"
                    logger.info(f"Successfully processed {pdf_file} -> {md_filename}")
                    
                except Exception as e:
                    error_msg = f"Error processing {pdf_file}: {str(e)}"
                    results[pdf_file] = error_msg
                    logger.error(error_msg)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing folder {input_folder}: {str(e)}")
            return {"error": f"Failed to process folder: {str(e)}"}

# Convenience functions for direct usage
def parse_pdf_to_markdown(pdf_path: str) -> str:
    """
    Convenience function to parse a single PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Parsed markdown text
    """
    client = PDFParsingClient()
    return client.parse_single_pdf(pdf_path)

def parse_pdf_folder_to_markdown(input_folder: str, 
                                output_folder: str = "parsed_markdown",
                                overwrite: bool = False) -> Dict[str, str]:
    """
    Convenience function to parse all PDFs in a folder.
    
    Args:
        input_folder (str): Path to folder with PDF files
        output_folder (str): Path to output folder for markdown files
        overwrite (bool): Whether to overwrite existing files
        
    Returns:
        Dict[str, str]: Processing results for each file
    """
    client = PDFParsingClient()
    return client.parse_pdf_folder(input_folder, output_folder, overwrite)

# Example usage
if __name__ == "__main__":
    # Example 1: Parse single PDF
    try:
        client = PDFParsingClient()
        markdown_text = client.parse_single_pdf("example.pdf")
        print("Single PDF parsed successfully!")
        print(f"Markdown length: {len(markdown_text)} characters")
    except Exception as e:
        print(f"Error parsing single PDF: {e}")
    
    # Example 2: Parse folder of PDFs
    try:
        results = client.parse_pdf_folder(
            input_folder="./pdf_files",
            output_folder="./markdown_files",
            overwrite=False
        )
        print("\nFolder processing results:")
        for pdf_file, status in results.items():
            print(f"  {pdf_file}: {status}")
    except Exception as e:
        print(f"Error parsing PDF folder: {e}")