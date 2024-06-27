FABRIC = ["COTTON", "POLYESTER", "ACRYLIC", "NYLON", "RAYON", "SPAN", "LINEN", "POLYURETAN", "MODAL", "WOOL", "TENCEL", "ACETATE"]

class PortionChanger:
    @staticmethod
    def portion_to_str(portion):
        result = []

        for i in range(len(portion)):
            if portion[i] > 0:
                result.append(f"{FABRIC[i]} {portion[i] * 100:.0f}")

        return " ".join(result)

    @staticmethod
    def str_to_portion(input_str):
        portion = [0.0] * 12

        # 문자열을 데이터 형태로 변환하는 코드
        input_list = input_str.split()
        for i in range(0, len(input_list), 2):
            material = input_list[i]
            percentage = float(input_list[i+1]) / 100
            index = FABRIC.index(material)
            portion[index] = percentage

        return portion