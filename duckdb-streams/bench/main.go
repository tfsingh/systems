package main

import (
	"bytes"
	"log"
	"net/http"
	"sync"
	"time"
)

func sendRequest(client *http.Client, url, contentType string, payload []byte) {
	req, err := http.NewRequest("POST", url, bytes.NewBuffer(payload))
	if err != nil {
		log.Println("Error creating request:", err)
		return
	}

	req.Header.Set("Content-Type", contentType)

	resp, err := client.Do(req)
	if err != nil {
		//log.Println(err)
		return
	}
	defer resp.Body.Close()

	//log.Println(resp.Status)
}

func requestWorker(client *http.Client, url, contentType string, payload1, payload2 []byte, duration time.Duration, wg *sync.WaitGroup, results chan<- float64) {
	defer wg.Done()

	startTime := time.Now()
	reqCount := 0
	endTime := startTime.Add(duration)

	for time.Now().Before(endTime) {
		sendRequest(client, url, contentType, payload1)
		sendRequest(client, url, contentType, payload2)
		reqCount += 2
	}

	elapsedTime := time.Since(startTime)
	elapsedSeconds := elapsedTime.Seconds()

	averageRequestsPerSecond := float64(reqCount) / elapsedSeconds
	results <- averageRequestsPerSecond
}

func main() {
	client := &http.Client{}
	url := "http://localhost:8080"
	contentType := "application/json"
	payload1 := []byte(`{"stream_id": "stream_1", "data": {"user_id": 1, "clicks": 1}}`)
	payload2 := []byte(`{"stream_id": "stream_2", "data": {"user_id": 1, "purchases": 1}}`)
	duration := 20 * time.Second

	var wg sync.WaitGroup
	numGoroutines := 4

	results := make(chan float64, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go requestWorker(client, url, contentType, payload1, payload2, duration, &wg, results)
	}

	wg.Wait()
	close(results)

	var total float64
	for result := range results {
		total += result
	}

	log.Printf("Sum of average requests per second: %.2f\n", total)
}
