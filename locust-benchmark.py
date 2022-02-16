import random
from locust import HttpUser, between, task

class WebsiteUser(HttpUser):
    wait_time = between(0.2, 1.0)

    def on_start(self):
        self.uid = str(random.randint(0, 100_000)).zfill(6)

    @task
    def getter(self):
        self.client.get("/")
        
    @task
    def simulate1(self):
        self.client.post("/predict/", json={
            "text": "kiekt 'm gaan jongh"
        })
    
    @task
    def simulate2(self):
        self.client.post("/sk_predict/", json={
            "text": "kiekt 'm gaan jongh"
        })
