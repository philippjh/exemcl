#ifndef CSVFILE_H_
#define CSVFILE_H_

#include <Eigen/Eigen>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

class CSVFile {
public:
    CSVFile() = default;

    /**
     * Returns the amount of columns currently present in the CSVFile.
     *
     * @return The amount of columns.
     */
    int getColumnCount() {
        return _columns.size();
    };

    /**
     * Returns the amount of rows currently present in the CSVFile.
     *
     * @return The amount of rows.
     */
    int getRowCount() {
        return _rows.size();
    };

    /**
     * Updates the column header names. This operation will clear all rows from the CSVFile.
     * @param columns The new set of column names.
     */
    void setColumns(std::vector<std::string> columns) {
        _columns = std::move(columns);
        _rows.clear();
    };

    /**
     * Adds a new row to the CSVFile.
     * @param row A vector of string values representing the row.
     * @return The row index of the newly created row.
     */
    void addRow(unsigned int index, std::vector<std::string> row) {
        if (row.size() == _columns.size()) {
            if (_rows.find(index) == _rows.end())
                _rows.emplace(index, row);
            else
                throw std::runtime_error("The specified row index is already present in the CSV file.");
        } else
            throw std::runtime_error("The specified row is either providing too much or an insufficient amount of columns.");
    };

    /**
     * Returns a matrix representation of the CSVFile.
     */
    Eigen::MatrixXd asMatrix() {
        int n = getRowCount();
        int m = getColumnCount();
        Eigen::MatrixXd resultMatrix(n, m);

        unsigned int rowID = 0;
        for (auto& rowPair : _rows) {
            for (int j = 0; j < m; j++) {
                try {
                    resultMatrix(rowID, j) = std::stod(rowPair.second[j]);
                } catch (std::exception& exception) { throw std::runtime_error("FeatureVectorSet::readFromFile: Could not convert a row contents to a floating-point number."); }
            }
            rowID += 1;
        }
        return resultMatrix;
    };

    virtual ~CSVFile() = default;

    static std::unique_ptr<CSVFile> readCSVFile(const std::string& fileName, char delimiter, bool header = true, const std::string& indexColumn = "ObjectID") {
        std::ifstream inputStream(fileName.c_str());
        std::unique_ptr<CSVFile> fileObject = std::make_unique<CSVFile>();

        bool firstLine = true;
        int indexColumnPos = -1;
        int currentIndex = -1;

        // Read file line by line.
        std::string currentLine;
        while (std::getline(inputStream, currentLine)) {
            std::vector<std::string> row;
            std::istringstream stringStream(currentLine);
            std::string token;

            while (std::getline(stringStream, token, delimiter))
                row.push_back(token);

            if (firstLine) {
                firstLine = false;
                if (header) {
                    // If we expect a header, then we will try to read it.
                    // Check for duplicates in column names.
                    std::set<std::string> headerDupCheck;
                    for (auto& s : row) {
                        if (headerDupCheck.find(s) == headerDupCheck.end())
                            headerDupCheck.insert(s);
                        else
                            throw std::runtime_error("Header contains duplicate column names.");
                    }

                    // Search for the index column.
                    auto indexColIt = std::find(row.begin(), row.end(), indexColumn);
                    indexColumnPos = static_cast<int>(std::distance(row.begin(), indexColIt));
                    if (indexColumnPos == row.size())
                        indexColumnPos = -1;
                    else
                        row.erase(indexColIt);

                    // Set the column names.
                    fileObject->setColumns(row);
                } else {
                    // Otherwise, we will generate a header.
                    std::vector<std::string> generatedHeaderNames;
                    generatedHeaderNames.reserve(row.size());
                    for (int i = 0; i < row.size(); i++)
                        generatedHeaderNames.push_back("C" + std::to_string(i));
                    fileObject->setColumns(generatedHeaderNames);

                    // The row, that we have read, will be added as a regular row.
                    currentIndex = 0;
                    fileObject->addRow(static_cast<unsigned int>(currentIndex), row);
                }
            } else {
                if (indexColumnPos != -1) {
                    currentIndex = std::stoi(row[indexColumnPos]);
                    row.erase(row.begin() + indexColumnPos);
                } else
                    currentIndex += 1;

                if (row.size() == fileObject->getColumnCount())
                    fileObject->addRow(static_cast<unsigned int>(currentIndex), row);
                else
                    throw std::runtime_error("The column lengths are differing in the provided csv file.");
            }
        }

        return fileObject;
    };

private:
    std::vector<std::string> _columns;
    std::map<unsigned int, std::vector<std::string>> _rows;
};

#endif /* CSVFILE_H_ */