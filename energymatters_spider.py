import scrapy
import re

class EnergySpider(scrapy.Spider):
    name = "energymatters"
    start_urls = ['https://www.energymatters.com.au/solar-location/williamtown-2318/']  # Restart after broken link
    # start_urls = ['https://www.energymatters.com.au/solar-location/williamtown-2318/']  # Restart after broken link
    # start_urls = ['https://www.energymatters.com.au/solar-location/valencia-creek-3860/']  # Restart after broken link
    # start_urls = ['https://www.energymatters.com.au/solar-location/unley-5061/']  # Restart after broken link
    # start_urls = ['https://www.energymatters.com.au/solar-location/reids-creek-4625/']  # Restart after broken link
    # start_urls = ['https://www.energymatters.com.au/solar-location/oak-flats-2529/']  # Restart after broken link
    # start_urls = ['https://www.energymatters.com.au/solar-location/kangaroo-creek-2460/']  # Restart after broken link
    # start_urls = ['https://www.energymatters.com.au/solar-location/homebush-west-2140/']  # Restart after broken link
    # start_urls = ['https://www.energymatters.com.au/solar-location/gordon-park-4031/']  # Restart after broken link
    # start_urls = ['https://www.energymatters.com.au/solar-location/daintree-4873/']  # Restart after broken link
    # start_urls = ['https://www.energymatters.com.au/solar-location/beaumont-2577/'] # Restart after broken link
    # start_urls = ['https://www.energymatters.com.au/solar-location/abbotsford-2046/'] # First page

    def parse(self, response):
        postcode_string = response.xpath('//meta[@property=$val]', val='og:url').extract()
        postcode_digits = re.findall(r'\d+', str(postcode_string))
        postcode = 'POA' + str(postcode_digits)
        yield{ postcode: response.css('li::text').re(r'(irradiation[a-z|\s|A-Z|\d+|\.|:|/]{25})')}

        next_page = response.xpath('//a[@rel=$val]/@href', val='next').extract_first()
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)




