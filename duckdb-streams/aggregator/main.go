package main

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/apple/foundationdb/bindings/go/src/fdb"
	"github.com/apple/foundationdb/bindings/go/src/fdb/directory"
	"github.com/apple/foundationdb/bindings/go/src/fdb/tuple"
	"github.com/google/uuid"
	"github.com/marcboeker/go-duckdb"
)

var (
	founddb                    fdb.Database
	ddb                        *sql.DB
	con                        driver.Conn
	streamOneDir, streamTwoDir directory.DirectorySubspace
)

var (
	streamOneId      = "stream_1"
	streamTwoId      = "stream_2"
	jsonInstallation = "INSTALL json; LOAD json;"
	startByte        = "\x00"
	endByte          = "\xff"
	joinQuery        = `	SELECT 
			s1.user_id,
			SUM(s1.clicks) AS total_clicks,
			SUM(s2.purchases) AS total_purchases
		FROM 
			stream_1_data s1
		JOIN 
			stream_2_data s2 ON s1.user_id = s2.user_id
		GROUP BY 
			s1.user_id;`
	dropStreamOne = "DROP TABLE IF EXISTS stream_1_data"
	dropStreamTwo = "DROP TABLE IF EXISTS stream_2_data"
)

func main() {
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

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		go cycle()
	}
}

func cycle() {
	createTable(streamOneId, streamOneDir)
	createTable(streamTwoId, streamTwoDir)

	defer func() {
		_, err := ddb.Exec(dropStreamOne)
		if err != nil {
			log.Println(err)
		}
		_, err = ddb.Exec(dropStreamTwo)
		if err != nil {
			log.Println(err)
		}
	}()

	rows, err := ddb.Query(joinQuery)
	if err != nil {
		log.Println(err)
		return
	}
	defer rows.Close()

	for rows.Next() {
		var userID, clicks, purchases interface{}

		if err := rows.Scan(&userID, &clicks, &purchases); err != nil {
			log.Println(err)
			return
		}

		fmt.Printf("User ID: %d, Clicks: %d, Purchases: %d\n", userID, clicks, purchases)
	}

	if err := rows.Err(); err != nil {
		log.Println("Error iterating through rows:", err)
	}
}

func createTable(stream string, stream_dir directory.DirectorySubspace) {
	var data [][]byte

	startKey := stream_dir.Pack(tuple.Tuple{fdb.Key(startByte)})
	endKey := stream_dir.Pack(tuple.Tuple{fdb.Key(endByte)})
	keyRange := fdb.KeyRange{Begin: startKey, End: endKey}

	// Fetch all the records from FDB for each directory
	_, err := founddb.Transact(func(tr fdb.Transaction) (interface{}, error) {
		res := tr.GetRange(keyRange, fdb.RangeOptions{})

		for kv := res.Iterator(); kv.Advance(); {
			kvp := kv.MustGet()
			data = append(data, kvp.Value)
		}

		// Wipe data to prevent late joins (perhaps not ideal behavior)
		tr.ClearRange(keyRange)

		return nil, nil
	})

	if err != nil {
		log.Println(err)
		return
	}

	// In order to prevent unmarshalling every record and coalescing
	// the subsequent marshall/file write of all records, we take
	// a piecemeal approach
	var dataSerialized []byte
	dataSerialized = append(dataSerialized, '[')
	for i, value := range data {
		dataSerialized = append(dataSerialized, value...)
		if i < len(data)-1 {
			dataSerialized = append(dataSerialized, ',')
		}
	}
	dataSerialized = append(dataSerialized, ']')

	// Dump the data to a file
	id := uuid.New()
	fileName := fmt.Sprintf("%s.json", id.String())
	err = os.WriteFile(fileName, dataSerialized, 0644)
	if err != nil {
		log.Println(err)
		return
	}

	defer func() {
		err = os.Remove(fileName)
		if err != nil {
			log.Println(err)
		}
	}()

	// Execute the query; DuckDB's read_json method infers the schema
	// from the json file
	if _, err := ddb.Exec(createTableQuery(stream, fileName)); err != nil {
		log.Println(err)
	}
}

func createTableQuery(stream string, filename string) string {
	return fmt.Sprintf(`CREATE TABLE %s_data AS SELECT * FROM read_json('%s');`, stream, filename)
}
