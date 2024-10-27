"""
from pyopensky.s3 import S3Client
from datetime import datetime

s3 = S3Client()

start_date = datetime.strptime("2022-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2022-11-19", "%Y-%m-%d")

for obj in s3.s3client.list_objects("competition-data", recursive=True):
    try:
        obj_date = datetime.strptime(obj.object_name.split('.')[0], "%Y-%m-%d")
    except ValueError:
        obj_date = None

    if obj_date is None or not (start_date <= obj_date <= end_date):
        print(f"{obj.bucket_name=}, {obj.object_name=}")
        s3.download_object(obj)
"""

from pyopensky.s3 import S3Client

s3 = S3Client()

for obj in s3.s3client.list_objects("competition-data", recursive=True):
    if obj.object_name == "final_submission_set.csv":
        print(f"{obj.bucket_name=}, {obj.object_name=}")
        s3.download_object(obj)
