package main

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/apple/foundationdb/bindings/go/src/fdb"
	"github.com/apple/foundationdb/bindings/go/src/fdb/directory"
	"github.com/apple/foundationdb/bindings/go/src/fdb/tuple"
	"github.com/google/uuid"
	"github.com/marcboeker/go-duckdb"
)

var (
	buffer                     map[string][]interface{}
	mu                         sync.Mutex
	founddb                    fdb.Database
	ddb                        *sql.DB
	con                        driver.Conn
	streamOneDir, streamTwoDir directory.DirectorySubspace
)

var (
	streamOneId      = "stream_1"
	streamTwoId      = "stream_2"
	invalidMethod    = "Only POST method is allowed"
	readError        = "Unable to read request body"
	invalidJson      = "Invalid JSON"
	emptyFilename    = "Empty filename"
	jsonInstallation = "INSTALL json; LOAD json;"
	streamOneAgg     = "SELECT user_id, SUM(clicks) as clicks FROM data GROUP BY user_id"
	streamTwoAgg     = "SELECT user_id, SUM(purchases) as purchases FROM data GROUP BY user_id"
	dropTable        = "DROP TABLE IF EXISTS data"
	startByte        = "\x00"
	endByte          = "\xff"
	joinQuery        = `SELECT
		s1.user_id,
		s1.clicks,
		s2.purchases
	FROM
		(SELECT user_id, SUM(clicks) AS clicks FROM stream_1_data GROUP BY user_id) s1
	JOIN
		(SELECT user_id, SUM(purchases) AS purchases FROM stream_2_data GROUP BY user_id) s2
	ON
		s1.user_id = s2.user_id;`
	dropStream1 = "DROP TABLE IF EXISTS stream_1_data"
	dropStream2 = "DROP TABLE IF EXISTS stream_2_data"
)

type Event struct {
	StreamID string      `json:"stream_id"`
	Data     interface{} `json:"data"`
}

func main() {
	buffer = make(map[string][]interface{})
	fdb.MustAPIVersion(620)
	founddb = fdb.MustOpenDefault()

	var err error
	_, err = founddb.Transact(func(tr fdb.Transaction) (interface{}, error) {
		streamOneDir, err = directory.CreateOrOpen(tr, []string{streamOneId}, nil)
		if err != nil {
			return nil, err
		}
		streamTwoDir, err = directory.CreateOrOpen(tr, []string{streamTwoId}, nil)
		if err != nil {
			return nil, err
		}
		return nil, nil
	})

	if err != nil {
		panic(err)
	}

	c, err := duckdb.NewConnector("", nil)
	if err != nil {
		panic(err)
	}

	con, err = c.Connect(context.Background())
	if err != nil {
		panic(err)
	}

	ddb = sql.OpenDB(c)
	defer ddb.Close()

	if _, err := ddb.Exec(jsonInstallation); err != nil {
		log.Println(err)
		return
	}

	go func() {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()

		for range ticker.C {
			go cycle()
		}
	}()

	http.HandleFunc("/", handleEvent)
	http.ListenAndServe(":8080", nil)
}

func cycle() {
	mu.Lock()
	defer mu.Unlock()

	var streamOneFile string
	var streamTwoFile string

	for streamId, data := range buffer {
		if len(data) < 1 {
			continue
		}
		dataSerialized, err := json.Marshal(data)
		if err != nil {
			log.Println(err)
			return
		}

		id := uuid.New()
		fileName := fmt.Sprintf("%s.json", id.String())
		err = os.WriteFile(fileName, dataSerialized, 0644)
		if err != nil {
			log.Println(err)
			return
		}

		if streamId == streamOneId {
			streamOneFile = fileName
		} else {
			streamTwoFile = fileName
		}
		buffer[streamId] = nil
	}

	defer func() {
		if streamOneFile != "" {
			err := os.Remove(streamOneFile)
			if err != nil {
				log.Println(err)
			}
		}
		if streamTwoFile != "" {
			err := os.Remove(streamTwoFile)
			if err != nil {
				log.Println(err)
			}
		}
	}()

	queryOneRes, queryOneErr := executeQuery(streamOneFile, streamOneAgg)
	queryTwoRes, queryTwoErr := executeQuery(streamTwoFile, streamTwoAgg)
	if queryOneErr != nil || queryTwoErr != nil {
		log.Println(queryOneErr, queryTwoErr)
	}

	_, err := founddb.Transact(func(tr fdb.Transaction) (interface{}, error) {
		if queryOneErr == nil {
			for _, data := range queryOneRes {
				tr.Set(streamOneDir.Pack(tuple.Tuple{fdb.Key(string(time.Now().String()))}), data)
			}
		}

		if queryTwoErr == nil {
			for _, data := range queryTwoRes {
				tr.Set(streamTwoDir.Pack(tuple.Tuple{fdb.Key(string(time.Now().String()))}), data)
			}
		}

		return nil, nil
	})

	if err != nil {
		log.Println(err)
		return
	}

}

func handleEvent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, invalidMethod, http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, readError, http.StatusBadRequest)
		return
	}

	var event Event
	err = json.Unmarshal(body, &event)
	if err != nil {
		http.Error(w, invalidJson, http.StatusBadRequest)
		return
	}

	mu.Lock()
	defer mu.Unlock()

	buffer[event.StreamID] = append(buffer[event.StreamID], &event.Data)

	w.WriteHeader(http.StatusOK)
}

func executeQuery(fileName string, query string) ([][]byte, error) {
	if fileName == "" {
		return nil, errors.New(emptyFilename)
	}
	new_q := createTableQuery(fileName)
	_, err := ddb.Exec(new_q)
	if err != nil {
		return nil, err
	}

	defer func() {
		_, err := ddb.Exec(dropTable)
		if err != nil {
			log.Println(err)
		}
	}()

	rows, err := ddb.Query(query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var result [][]byte
	for rows.Next() {
		columns, err := rows.Columns()
		if err != nil {
			return nil, err
		}

		values := make([]interface{}, len(columns))
		valuePtrs := make([]interface{}, len(columns))
		for i := range values {
			valuePtrs[i] = &values[i]
		}

		if err := rows.Scan(valuePtrs...); err != nil {
			return nil, err
		}

		rowMap := make(map[string]interface{})
		for i, colName := range columns {
			rowMap[colName] = values[i]
		}

		jsonData, err := json.Marshal(rowMap)
		if err != nil {
			return nil, err
		}
		result = append(result, jsonData)
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	return result, nil
}

func createTableQuery(fileName string) string {
	return fmt.Sprintf(`CREATE TABLE data AS SELECT * FROM read_json('%s')`, fileName)
}
