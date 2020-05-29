import unittest
from detection3d.vis.gen_html_report import *


class TestGenHtmlReport(unittest.TestCase):

  def setUp(self):
    self.string = 'test'
    self.number = 1

  def test_add_document_text(self):
    res = add_document_text(self.string, self.string)
    self.assertIsNotNone(res)

  # WriteToHtmlReportFile(document_text, analysis_text, html_report_path, width)
  def test_write_to_html_report_file(self):
    res = write_html_report_for_single_landmark(self.string, self.string, self.string, self.number)
    self.assertIsNone(res)


if __name__ == '__main__':
  unittest.main()