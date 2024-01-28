import os
import scrapy
from readability.readability import Document
import html2text

class VbtProSpider(scrapy.Spider):
    name = "vbt_pro"
    allowed_domains = ['vectorbt.pro']

    # Initialize the spider with the secret_url parameter
    def __init__(self, secret_url=None, *args, **kwargs):
        super(VbtProSpider, self).__init__(*args, **kwargs)
        self.secret_url = secret_url
        self.start_urls = [
            f'https://vectorbt.pro/{self.secret_url}/features/',
            f'https://vectorbt.pro/{self.secret_url}/tutorials/',
            f'https://vectorbt.pro/{self.secret_url}/documentation/',
            f'https://vectorbt.pro/{self.secret_url}/api/',
            f'https://vectorbt.pro/{self.secret_url}/cookbook/'
        ]

        # Combine and centralize data directories from both spiders
        self.base_dir = 'docs/vbt_pro'
        self.api_dir = os.path.join(self.base_dir, 'api')

    def start_requests(self):
        # Ensure base directories exist
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.api_dir, exist_ok=True)

        for url in self.start_urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        print(f'Parsing {response.url}')
        # Decode the response body explicitly with UTF-8 if necessary
        response_body = response.body.decode('utf-8', errors='replace')

        # Using readability to extract the main content
        document = Document(response_body)
        summary = document.summary()

        # Converting HTML summary to Markdown using html2text
        converter = html2text.HTML2Text()
        markdown_content = converter.handle(summary)

        # Split the URL and create a filename based on the number of segments
        url_segments = response.url.split('/')
        # Filter out empty segments and take segments after the domain
        relevant_segments = [segment for segment in url_segments[4:] if segment]

        # Construct the filename dynamically
        if relevant_segments[0] == 'api':
            # Special handling for API section
            filename = os.path.join(self.api_dir, f'{relevant_segments[-1]}.txt')
        else:
            # Join segments with a dash to form the filename
            article_name = '-'.join(relevant_segments) + '.txt'
            filename = os.path.join(self.base_dir, article_name)

        # Write main content as Markdown with UTF-8 encoding
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        # Follow links that start with any of the start URLs
        for href in response.css('a::attr(href)').getall():
            full_href = response.urljoin(href)
            if any(full_href.startswith(start_url) for start_url in self.start_urls):
                yield response.follow(href, self.parse)

