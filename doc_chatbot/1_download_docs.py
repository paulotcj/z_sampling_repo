# you may need to install: pip install pdfplumber

import os
import requests
import re
from typing import Match
import pdfplumber
import string
import logging

#-------------------------------------------------------------------------
class CleanDownloadDir : 
    #-------------------------------------------------------------------------
    @staticmethod
    def clean(dir: str) -> bool:

        try:
            if not os.path.exists(dir):

                logging.info(f"Directory does not exist: {dir}")
                return False
            for filename in os.listdir(dir):
                file_path: str = os.path.join(dir, filename)

                if os.path.isfile(path = file_path) :
                    os.remove(path = file_path)
            return True
        except Exception as e:
            logging.info(f"Error cleaning directory {dir}: {e}")
            return False
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class DownloadDocuments :
    #-------------------------------------------------------------------------
    @staticmethod
    def download_pdf_from_url(pdf_url: str, save_dir: str) -> str:

        if not os.path.exists(path = save_dir):
            os.makedirs(name = save_dir)

        response: requests.Response = requests.get(url = pdf_url)
        response.raise_for_status()

        filename: str|None = None

        #------
        # try to get filename from header
        content_disposition: str|None = response.headers.get(key = 'Content-Disposition')
        match: Match[str]|None = None

        if content_disposition:
            match = re.search(pattern = r'filename="?([^"]+)"?', string = content_disposition)
            if match:
                filename = match.group(1)
        #------

        # url basename is missing
        if not filename:
            filename = os.path.basename(p = pdf_url )
            if not filename.lower().endswith('.pdf'):
                filename += '.pdf'

        save_path: str = os.path.join(save_dir, filename)
        with open(file = save_path, mode = 'wb') as f:
            f.write(response.content)

        return save_path
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class ConvertPDF :   
    PRINTABLE : set[str] = set(string.printable)
    #-------------------------------------------------------------------------
    @staticmethod
    def clean_text(text: str) -> str:
        # remove non-printable chars
        return ''.join(filter(lambda x: x in ConvertPDF.PRINTABLE, text))    
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    @staticmethod
    def to_markdown(pdf_filepath: str, md_dir: str, link : str) -> None:

        if not os.path.exists(path = md_dir):
            os.makedirs(name = md_dir)

        pdf_basename : str = os.path.basename(p = pdf_filepath)
        md_filename  : str = os.path.splitext(p = pdf_basename)[0] + ".md"
        md_path      : str = os.path.join(md_dir, md_filename)


        #----------------------------------------
        with pdfplumber.open(pdf_filepath) as pdf:
            all_text_list : list[str] = []
            page: pdfplumber.page.Page

            if link :
                all_text_list.append( f"Source Link: {link}\n\n" )

            #----------------------------------------
            for page in pdf.pages:
                page_text: str | None = page.extract_text()
                if page_text is not None:
                    page_text = ConvertPDF.clean_text(text = page_text)
                    # page_text = clean_text(page_text)
                    all_text_list.append(page_text)

                # page separator - looks like a good idea to keep it similar to
                #  the original PDF
                all_text_list.append("\n\n---\n\n")
            #----------------------------------------

            if link :
                all_text_list.append( f"Source Link: {link}\n\n" )
        #----------------------------------------

        all_text : str = ''.join(all_text_list)

        # remove excessive separators and trailing whitespace
        all_text = re.sub(r"(\n\n---\n\n)+", "\n\n---\n\n", all_text)
        all_text = all_text.strip()

        with open(md_path, "w", encoding="utf-8") as md_file:
            md_file.write(all_text)
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def main():
    
    manuals_to_download : list[str] = [
        "https://support.ninjakitchen.com/hc/en-us/article_attachments/5044328878236",
        "https://support.ninjakitchen.com/hc/en-us/article_attachments/7133551739676",
        "https://support.ninjakitchen.com/hc/en-us/article_attachments/14874907931420",
        "https://support.ninjakitchen.com/hc/en-us/article_attachments/5204932314268",
    ]

    save_pdf_path : str = "./pdf/manuals/"
    save_md_path : str  = "./md/manuals/"

    links_and_filename : list[tuple[str,str]] = []


    CleanDownloadDir.clean(dir = save_pdf_path)
    CleanDownloadDir.clean(dir = save_md_path)

    #----------------------------------------
    for link in manuals_to_download :
        saved_filename: str = DownloadDocuments.download_pdf_from_url(
            pdf_url     = link ,
            save_dir    = save_pdf_path
        )

        if saved_filename :
            links_and_filename.append( (link, saved_filename) )
    #----------------------------------------

    for e in links_and_filename:
        file_link : str = e[0]
        pdf_file : str = e[1]

        ConvertPDF.to_markdown(pdf_filepath = pdf_file , md_dir = save_md_path, link = file_link )
    #----------------------------------------

#-------------------------------------------------------------------------


if __name__ == "__main__":
    main()

